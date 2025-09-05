// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Regex expressions
use arrow::array::{Array, ArrayRef, AsArray, StringViewBuilder};
use arrow::compute::kernels::regexp;
use arrow::datatypes::DataType;
use datafusion_common::cast::as_int64_array;
use datafusion_common::exec_err;
use datafusion_common::ScalarValue;
use datafusion_common::{arrow_datafusion_err, plan_err};
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{ColumnarValue, Documentation, TypeSignature};
use datafusion_expr::{ScalarUDFImpl, Signature, Volatility};
use datafusion_macros::user_doc;
use std::any::Any;
use std::sync::Arc;

#[user_doc(
    doc_section(label = "Regular Expression Functions"),
    description = "Matches a regular expression against a string and extracts a specific match group.",
    syntax_example = "regexp_extract(str, regexp, idx)",
    sql_example = r#"```sql
            > select regexp_extract('bd', '(b|d)(b|d)', 1);
            +--------------------------------------------------------+
            | regexp_extract(Utf8("bd"),Utf8("(b|d)(b|d)"),Int64(1)) |
            +--------------------------------------------------------+
            | b                                                      |
            +--------------------------------------------------------+

            > select regexp_extract('bd', '(b|d)(b|d)', 2);
            +--------------------------------------------------------+
            | regexp_extract(Utf8("bd"),Utf8("(b|d)(b|d)"),Int64(2)) |
            +--------------------------------------------------------+
            | d                                                      |
            +--------------------------------------------------------+
```
Additional examples can be found [here](https://github.com/apache/datafusion/blob/main/datafusion-examples/examples/regexp.rs)
"#,
    standard_argument(name = "str", prefix = "String"),
    argument(
        name = "regexp",
        description = "Regular expression to match against.
            Can be a constant, column, or function."
    ),
    argument(
        name = "idx",
        description = "Group match index, 1-based.
            Can be a constant, column, or function."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct RegexpExtractFunc {
    signature: Signature,
}

impl Default for RegexpExtractFunc {
    fn default() -> Self {
        Self::new()
    }
}

impl RegexpExtractFunc {
    pub fn new() -> Self {
        use DataType::*;
        Self {
            signature: Signature::one_of(
                vec![
                    // Int64 and not UInt64 because idx=0 in SQL binds to Int64
                    TypeSignature::Exact(vec![Utf8View, Utf8View, Int64]),
                    TypeSignature::Exact(vec![Utf8, Utf8, Int64]),
                    TypeSignature::Exact(vec![LargeUtf8, LargeUtf8, Int64]),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for RegexpExtractFunc {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "regexp_extract"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(match &arg_types[0] {
            DataType::Null => DataType::Null,
            other => other.clone(),
        })
    }

    fn invoke_with_args(
        &self,
        args: datafusion_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        // determine if dealing with scalar-only inputs or at least one array input
        let args = &args.args;
        // len is None if all scalars, Some(len) if any array exists
        let len = args
            .iter()
            .fold(Option::<usize>::None, |acc, arg| match arg {
                ColumnarValue::Scalar(_) => acc,
                ColumnarValue::Array(a) => Some(a.len()),
            });

        // normalize inputs
        let is_scalar = len.is_none();
        let inferred_length = len.unwrap_or(1);
        // broadcast scalars to inferred length (1 for scalar-only)
        let args = args
            .iter()
            .map(|arg| arg.to_array(inferred_length))
            .collect::<Result<Vec<_>>>()?;

        // preserve input semantics
        // scalar inputs -> scalar outputs
        // array inputs -> array outputs
        let result = regexp_extract(&args);
        if is_scalar {
            // If all inputs are scalar, keeps output as scalar
            let result = result.and_then(|arr| ScalarValue::try_from_array(&arr, 0));
            result.map(ColumnarValue::Scalar)
        } else {
            result.map(ColumnarValue::Array)
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

pub fn regexp_extract(args: &[ArrayRef]) -> Result<ArrayRef> {
    match args.len() {
        3 => {
            let group_indices = as_int64_array(&args[2])?;

            let mut res = StringViewBuilder::with_capacity(group_indices.len());

            let results = regexp::regexp_match(&args[0], &args[1], None)
                .map_err(|e| arrow_datafusion_err!(e))?;

            for (match_groups, idx_1based) in
                results.as_list::<i32>().iter().zip(group_indices.iter())
            {
                match (match_groups, idx_1based) {
                    (Some(group), Some(idx)) => {
                        let idx_0based: Result<usize, _> = (idx - 1).try_into();
                        match idx_0based {
                            Ok(idx_0based) if idx_0based < group.len() => {
                                let group = match group.data_type() {
                                    DataType::Utf8View => {
                                        group.as_string_view().value(idx_0based)
                                    }
                                    DataType::Utf8 => {
                                        group.as_string::<i32>().value(idx_0based)
                                    }
                                    DataType::LargeUtf8 => {
                                        group.as_string::<i64>().value(idx_0based)
                                    }
                                    e => {
                                        return plan_err!("regexp_extract was called with unexpected data type {e:?}");
                                    }
                                };
                                res.append_value(group);
                            }
                            _ => {
                                // idx-1 doesn't map to usize or is out-of-bounds for the match result
                                res.append_null();
                            }
                        }
                    }
                    _ => {
                        // match result or result are missing (NULL)
                        res.append_null();
                    }
                }
            }
            let res = res.finish();

            Ok(Arc::new(res))
        }
        other => exec_err!(
            "regexp_extract was called with {other} arguments. It requires exactly 3."
        ),
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::regexpextract::regexp_extract;
    use arrow::array::{Int64Array, StringViewBuilder};
    use std::sync::Arc;

    #[test]
    fn test_regexp_extract() {
        let mut value_builder = StringViewBuilder::new();
        let mut pattern_builder = StringViewBuilder::new();
        let mut idx_builder = Vec::<Option<i64>>::new();
        let mut expected_builder = StringViewBuilder::new();

        let tests = vec![
            // tuple format: input string, regex pattern, match group index, expected
            // None is translated to missing (NULL)
            (Some("bd"), Some("(b|d)(b|d)"), Some(1), Some("b")), // positive test: 2 matches, extract first group
            (Some("bd"), Some("(b|d)(b|d)"), Some(2), Some("d")), // positive test: 2 matches, extract second group
            (Some("bd"), Some("(b|d)(b|d)"), Some(3), None), // negative test: 2 matches, extract 3rd
            (Some("bd"), Some("(b|d)(b|d)"), Some(0), None), // negative test: 2 matches, extract 0th for 1-based indexing
            (Some("ae"), Some("(b|d)(b|d)"), Some(1), None), // negative test: 0 matches, extract 1st
            (None, Some("(b|d)(b|d)"), Some(1), None), // negative test: missing input string
            (Some("bd"), None, Some(1), None), // negative test: missing regex pattern
            (Some("bd"), Some("(b|d)(b|d)"), None, None), // negative test: missing group idx
        ];

        for (value, pattern, idx, expected) in tests {
            match value {
                Some(value) => value_builder.append_value(value),
                None => value_builder.append_null(),
            }
            match pattern {
                Some(pattern) => pattern_builder.append_value(pattern),
                None => pattern_builder.append_null(),
            }
            idx_builder.push(idx);
            match expected {
                Some(expected) => expected_builder.append_value(expected),
                None => expected_builder.append_null(),
            }
        }

        let values = value_builder.finish();
        let patterns = pattern_builder.finish();
        let idx = Int64Array::from(idx_builder);
        let expected = expected_builder.finish();

        let re = regexp_extract(&[Arc::new(values), Arc::new(patterns), Arc::new(idx)])
            .unwrap();

        assert_eq!(re.as_ref(), &expected);
    }
}

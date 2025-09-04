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
use arrow::array::{Array, ArrayRef, AsArray};
use arrow::compute::kernels::regexp;
use arrow::datatypes::DataType;
use arrow::datatypes::Field;
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
    description = "Returns the first [regular expression](https://docs.rs/regex/latest/regex/#syntax) matches in a string.",
    syntax_example = "regexp_extract(str, regexp, id)",
    sql_example = r#"```sql
            > select regexp_extract('Köln', '[a-zA-Z]ö[a-zA-Z]{2}');
            +---------------------------------------------------------+
            | regexp_extract(Utf8("Köln"),Utf8("[a-zA-Z]ö[a-zA-Z]{2}")) |
            +---------------------------------------------------------+
            | [Köln]                                                  |
            +---------------------------------------------------------+
            SELECT regexp_extract('aBc', '(b|d)', 'i');
            +---------------------------------------------------+
            | regexp_extract(Utf8("aBc"),Utf8("(b|d)"),Utf8("i")) |
            +---------------------------------------------------+
            | [B]                                               |
            +---------------------------------------------------+
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
        name = "id",
        description = r#"Optional regular expression flags that control the behavior of the regular expression. The following flags are supported:
  - **i**: case-insensitive: letters match both upper and lower case
  - **m**: multi-line mode: ^ and $ match begin/end of line
  - **s**: allow . to match \n
  - **R**: enables CRLF mode: when multi-line mode is enabled, \r\n is used
  - **U**: swap the meaning of x* and x*?"#
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

    // like regexp_match
    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(match &arg_types[0] {
            DataType::Null => DataType::Null,
            other => DataType::List(Arc::new(Field::new_list_field(other.clone(), true))),
        })
    }

    fn invoke_with_args(
        &self,
        args: datafusion_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let args = &args.args;
        let len = args
            .iter()
            .fold(Option::<usize>::None, |acc, arg| match arg {
                ColumnarValue::Scalar(_) => acc,
                ColumnarValue::Array(a) => Some(a.len()),
            });

        let is_scalar = len.is_none();
        let inferred_length = len.unwrap_or(1);
        let args = args
            .iter()
            .map(|arg| arg.to_array(inferred_length))
            .collect::<Result<Vec<_>>>()?;

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
            regexp::regexp_match(&args[0], &args[1], None)
                .map_err(|e| arrow_datafusion_err!(e))
        }
        other => exec_err!(
            "regexp_extract was called with {other} arguments. It requires exactly 3."
        ),
    }
}
#[cfg(test)]
mod tests {
    use crate::regex::regexpextract::regexp_extract;
    use arrow::array::StringArray;
    use arrow::array::{GenericStringBuilder, ListBuilder};
    use std::sync::Arc;

    #[test]
    fn test_case_sensitive_regexp_match2() {
        let values = StringArray::from(vec!["abc"; 5]);
        let patterns =
            StringArray::from(vec!["^(a)", "^(A)", "(b|d)", "(B|D)", "^(b|c)"]);

        let elem_builder: GenericStringBuilder<i32> = GenericStringBuilder::new();
        let mut expected_builder = ListBuilder::new(elem_builder);
        expected_builder.values().append_value("a");
        expected_builder.append(true);
        expected_builder.append(false);
        expected_builder.values().append_value("b");
        expected_builder.append(true);
        expected_builder.append(false);
        expected_builder.append(false);
        let expected = expected_builder.finish();

        let re = regexp_match(&[Arc::new(values), Arc::new(patterns)]).unwrap();

        assert_eq!(re.as_ref(), &expected);
    }

    #[test]
    fn test_case_insensitive_regexp_match2() {
        let values = StringArray::from(vec!["abc"; 5]);
        let patterns =
            StringArray::from(vec!["^(a)", "^(A)", "(b|d)", "(B|D)", "^(b|c)"]);
        let flags = StringArray::from(vec!["i"; 5]);

        let elem_builder: GenericStringBuilder<i32> = GenericStringBuilder::new();
        let mut expected_builder = ListBuilder::new(elem_builder);
        expected_builder.values().append_value("a");
        expected_builder.append(true);
        expected_builder.values().append_value("a");
        expected_builder.append(true);
        expected_builder.values().append_value("b");
        expected_builder.append(true);
        expected_builder.values().append_value("b");
        expected_builder.append(true);
        expected_builder.append(false);
        let expected = expected_builder.finish();

        let re = regexp_match(&[Arc::new(values), Arc::new(patterns), Arc::new(flags)])
            .unwrap();

        assert_eq!(re.as_ref(), &expected);
    }

    #[test] 
    fn test_unsupported_global_flag_regexp_match2() {
        let values = StringArray::from(vec!["abc"]);
        let patterns = StringArray::from(vec!["^(a)"]);
        let flags = StringArray::from(vec!["g"]);

        let re_err =
            regexp_match(&[Arc::new(values), Arc::new(patterns), Arc::new(flags)])
                .expect_err("unsupported flag should have failed");

        assert_eq!(re_err.strip_backtrace(), "Error during planning: regexp_match() does not support the \"global\" option");
    }
}

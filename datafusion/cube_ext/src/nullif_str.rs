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

use arrow::array::Array;
use arrow::array::*;
use arrow::datatypes::DataType;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::ColumnarValue;
use std::sync::Arc;

macro_rules! null_if_equal {
    ($LEFT:expr, $RIGHT:expr, $TYPE:ident) => {{
        let left = $LEFT
            .as_any()
            .downcast_ref::<$TYPE>()
            .expect("failed to downcast array");
        let right = $RIGHT
            .as_any()
            .downcast_ref::<$TYPE>()
            .expect("failed to downcast array");

        let res = left
            .iter()
            .zip(right.iter())
            .map(|(l, r)| if l == r { None } else { l })
            .collect::<$TYPE>();

        Arc::new(res) as ArrayRef
    }};
}

pub fn nullif_func_str(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let (left_arr, right_arr) = match (&args[0], &args[1]) {
        (ColumnarValue::Scalar(lhs), ColumnarValue::Scalar(rhs)) => {
            (lhs.to_array(), rhs.to_array())
        }
        (ColumnarValue::Array(lhs), ColumnarValue::Array(rhs)) => {
            (lhs.clone(), rhs.clone())
        }
        // align
        (ColumnarValue::Array(lhs), ColumnarValue::Scalar(rhs)) => {
            (lhs.clone(), rhs.clone().to_array_of_size(lhs.len()))
        }
        (ColumnarValue::Scalar(lhs), ColumnarValue::Array(rhs)) => {
            (lhs.clone().to_array_of_size(rhs.len()), rhs.clone())
        }
    };

    if left_arr.data_type() != right_arr.data_type() {
        return Err(DataFusionError::NotImplemented(
            "both arguments have to have the same type".to_string(),
        ));
    }

    let res = match left_arr.data_type() {
        DataType::Utf8 => null_if_equal!(left_arr, right_arr, StringArray),
        DataType::LargeUtf8 => null_if_equal!(left_arr, right_arr, LargeStringArray),
        _ => {
            return Err(DataFusionError::NotImplemented(
                "nullif_str supports Utf8 and LargeUtf8 only".to_string(),
            ))
        }
    };

    Ok(ColumnarValue::Array(res))
}

#[cfg(test)]
mod tests {
    use datafusion_common::ScalarValue;

    use super::*;

    #[test]
    fn test_nullif_str_array() {
        let a = GenericStringArray::<i32>::from(vec![
            Some("1"),
            None,
            Some("2"),
            Some("3"),
            Some("4"),
        ]);
        let b = GenericStringArray::<i32>::from(vec![
            None,
            None,
            Some("2"),
            Some("a"),
            Some("b"),
        ]);

        let expected = GenericStringArray::<i32>::from(vec![
            Some("1"),
            None,
            None,
            Some("3"),
            Some("4"),
        ]);

        match nullif_func_str(&[
            ColumnarValue::Array(Arc::new(a) as ArrayRef),
            ColumnarValue::Array(Arc::new(b) as ArrayRef),
        ])
        .unwrap()
        {
            ColumnarValue::Array(arr) => {
                let we = arr
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("failed to downcast array");
                assert_eq!(&expected, we);
            }
            _ => panic!("test_nullif_str_array failed"),
        }

        let a =
            GenericStringArray::<i64>::from(vec![Some("FOO"), Some("BAR"), Some("KEK")]);

        let expected =
            GenericStringArray::<i64>::from(vec![Some("FOO"), Some("BAR"), None]);

        match nullif_func_str(&[
            ColumnarValue::Array(Arc::new(a) as ArrayRef),
            ColumnarValue::Scalar(ScalarValue::LargeUtf8(Some("KEK".to_string()))),
        ])
        .unwrap()
        {
            ColumnarValue::Array(arr) => {
                let we = arr
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("failed to downcast array");
                assert_eq!(&expected, we);
            }
            _ => panic!("test_nullif_str_array failed"),
        }
    }
}

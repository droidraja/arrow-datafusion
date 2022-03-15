use crate::error::DataFusionError;
use arrow::error::ArrowError;
use futures::future::FutureExt;
use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(PartialEq, Debug)]
pub struct PanicError {
    msg: String,
}

impl PanicError {
    pub fn new(msg: String) -> PanicError {
        PanicError { msg }
    }
}

impl From<PanicError> for ArrowError {
    fn from(error: PanicError) -> Self {
        ArrowError::ComputeError(format!("Panic: {}", error.msg))
    }
}

impl From<PanicError> for DataFusionError {
    fn from(error: PanicError) -> Self {
        DataFusionError::Internal(format!("Panic: {}", error.msg))
    }
}

pub fn try_with_catch_unwind<F, R>(f: F) -> Result<R, PanicError>
where
    F: FnOnce() -> R,
{
    let result = catch_unwind(AssertUnwindSafe(f));
    match result {
        Ok(x) => Ok(x),
        Err(e) => match e.downcast::<String>() {
            Ok(s) => Err(PanicError::new(*s)),
            Err(e) => match e.downcast::<&str>() {
                Ok(m1) => Err(PanicError::new(m1.to_string())),
                Err(_) => Err(PanicError::new("unknown cause".to_string())),
            },
        },
    }
}

pub async fn async_try_with_catch_unwind<F, R>(future: F) -> Result<R, PanicError>
where
    F: Future<Output = R>,
{
    let result = AssertUnwindSafe(future).catch_unwind().await;
    match result {
        Ok(x) => Ok(x),
        Err(e) => match e.downcast::<String>() {
            Ok(s) => Err(PanicError::new(*s)),
            Err(e) => match e.downcast::<&str>() {
                Ok(m1) => Err(PanicError::new(m1.to_string())),
                Err(_) => Err(PanicError::new("unknown cause".to_string())),
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic;

    #[test]
    fn test_try_with_catch_unwind() {
        assert_eq!(
            try_with_catch_unwind(|| "ok".to_string()),
            Ok("ok".to_string())
        );
        assert_eq!(
            try_with_catch_unwind(|| panic!("oops")),
            Err(PanicError::new("oops".to_string()))
        );
        assert_eq!(
            try_with_catch_unwind(|| panic!("oops{}", "ie")),
            Err(PanicError::new("oopsie".to_string()))
        );
    }

    #[tokio::test]
    async fn test_async_try_with_catch_unwind() {
        assert_eq!(
            async_try_with_catch_unwind(async { "ok".to_string() }).await,
            Ok("ok".to_string())
        );
        assert_eq!(
            async_try_with_catch_unwind(async { panic!("oops") }).await,
            Err(PanicError::new("oops".to_string()))
        );
        assert_eq!(
            async_try_with_catch_unwind(async { panic!("oops{}", "ie") }).await,
            Err(PanicError::new("oopsie".to_string()))
        );
    }
}

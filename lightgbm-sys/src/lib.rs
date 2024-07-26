#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::redundant_static_lifetimes)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::upper_case_acronyms)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::ffi::{c_char, c_void};

#[link(name = "c")]
extern "C" {
    pub fn LGBM_GetLastError() -> *const c_char;

    pub fn LGBM_DatasetCreateFromMat(
        flat_data: *const c_void,
        dtype: i32,
        data_length: i32,
        feature_length: i32,
        arg: i32,
        params: *const c_char,
        reference: *const c_void,
        handle: *mut DatasetHandle,
    ) -> i32;

    pub fn LGBM_DatasetSetField(
        handle: DatasetHandle,
        field_name: *const c_char,
        data: *const c_void,
        data_length: i32,
        dtype: i32,
    ) -> i32;

    pub fn LGBM_DatasetCreateFromFile(
        filename: *const c_char,
        parameters: *const c_char,
        reference: *const c_void,
        out: *mut DatasetHandle,
    ) -> i32;

    pub fn LGBM_DatasetFree(handle: DatasetHandle) -> i32;

    pub fn LGBM_BoosterCreateFromModelfile(
        filename: *const c_char,
        out: *mut i32,
        handle: *mut BoosterHandle,
    ) -> i32;

    pub fn LGBM_BoosterCreate(
        train_data: DatasetHandle,
        parameters: *const c_char,
        out: *mut BoosterHandle,
    ) -> i32;

    pub fn LGBM_BoosterUpdateOneIter(handle: BoosterHandle, out_num_iterations: *mut i32) -> i32;

    pub fn LGBM_BoosterGetNumClasses(handle: BoosterHandle, out: *mut i32) -> i32;

    pub fn LGBM_BoosterPredictForMat(
        handle: BoosterHandle,
        data: *const c_void,
        dtype: i32,
        nrow: i32,
        ncol: i32,
        predict_type: i32,
        parameters: i32,
        start_iteration: i32,
        num_iteration: i32,
        parameter: *const c_char,
        out_len: *mut i64,
        out_result: *mut f64,
    ) -> i32;

    pub fn LGBM_BoosterGetNumFeature(handle: BoosterHandle, out: *mut i32) -> i32;

    pub fn LGBM_BoosterGetFeatureNames(
        handle: BoosterHandle,
        feature_name_length: i32,
        num_feature_names: *mut i32,
        num_feature: u64,
        out_buffer_len: *mut i64,
        out_strs: *mut *mut c_char,
    ) -> i32;

    pub fn LGBM_BoosterFeatureImportance(
        handle: BoosterHandle,
        importance_type: i32,
        iteration: i32,
        out_result: *mut f64,
    ) -> i32;

    pub fn LGBM_BoosterSaveModel(
        handle: BoosterHandle,
        num_iteration: i32,
        feature_importance_type: i32,
        num_iteration_name: i32,
        filename: *const c_char,
    ) -> i32;

    pub fn LGBM_BoosterFree(handle: BoosterHandle) -> i32;
}

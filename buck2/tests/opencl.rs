use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_ALL};
use opencl3::program::Program;

const SOURCE: &str = r#"
__kernel void add_one(__global float *values) {
    const size_t index = get_global_id(0);
    values[index] += 1.0f;
}
"#;

#[test]
fn opencl_compiles_a_kernel() {
    let device_id = get_all_devices(CL_DEVICE_TYPE_ALL)
        .expect("enumerate OpenCL devices")
        .into_iter()
        .next()
        .expect("an OpenCL device");
    let context = Context::from_device(&Device::new(device_id)).expect("create OpenCL context");
    Program::create_and_build_from_source(&context, SOURCE, "").expect("compile OpenCL kernel");
}

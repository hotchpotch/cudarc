use cudarc::{
    driver::{CudaDevice, DriverError, Graph, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;

    // You can load a function from a pre-compiled PTX like so:
    dev.load_ptx(Ptx::from_file("./examples/sin.ptx"), "sin", &["sin_kernel"])?;

    // and then retrieve the function with `get_func`
    let f = dev.get_func("sin", "sin_kernel").unwrap();

    let a_host = [1.0, 2.0, 3.0];

    let a_dev = dev.htod_copy(a_host.into())?;
    let mut b_dev = a_dev.clone();

    let n = 3;
    let cfg = LaunchConfig::for_num_elems(n);
    let loops = 10_000;

    dev.synchronize()?;
    let start = std::time::Instant::now();
    for _ in 0..loops {
        unsafe { f.launch(cfg, (&mut b_dev, &a_dev, n as i32)) }?;
    }
    dev.synchronize()?;
    println!("Without cuda graphs {:?}", start.elapsed());
    // assert_eq!(&a_host, a_host_2.as_slice());

    let a_dev = dev.htod_copy(a_host.into())?;
    let mut b_dev = a_dev.clone();
    let handle = Graph::start_capture(&dev)?;
    for _ in 0..loops {
        unsafe { f.launch(cfg, (&mut b_dev, &a_dev, n as i32)) }.unwrap();
    }
    let graph = handle.end_capture()?;
    // First launch is supposed to be slower
    graph.launch()?;
    dev.synchronize()?;
    let start = std::time::Instant::now();
    graph.launch()?;
    dev.synchronize()?;
    println!("With cuda graphs {:?}", start.elapsed());
    // assert_eq!(&a_host, a_host_2.as_slice());

    Ok(())
}

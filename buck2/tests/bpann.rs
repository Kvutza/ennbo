use bpann::distance::l2_sq_f32;

#[test]
fn squared_l2_is_exact_for_small_vectors() {
    let left = [1.0, -2.0, 4.0, 0.5];
    let right = [-1.0, 1.0, 2.0, -0.5];
    assert_eq!(l2_sq_f32(&left, &right), 18.0);
}

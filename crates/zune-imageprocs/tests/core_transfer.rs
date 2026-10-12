use zune_core::colorspace::{ColorCharacteristics, ColorSpace};
use zune_image::{image::Image, traits::OperationsTrait};
use zune_imageprocs::resize::{Resize, ResizeMethod};

#[test]
fn unsupported_transfer_is_rejected_before_resize_mutates_pixels_or_metadata() {
    for transfer in [
        ColorCharacteristics::PQ,
        ColorCharacteristics::HLG,
        ColorCharacteristics::Unknown(255),
    ] {
        let pixels = [
            10, 20, 30, 100, 40, 50, 60, 150, 70, 80, 90, 200, 100, 110, 120, 250,
        ];
        let mut image = Image::from_u8(&pixels, 2, 2, ColorSpace::RGBA);
        image.metadata_mut().set_color_trc(transfer);
        let before_depth = image.depth();
        let before_alpha = image.metadata().is_premultiplied_alpha();
        assert!(Resize::new(3, 3, ResizeMethod::Bilinear)
            .execute_impl(&mut image)
            .is_err());
        assert_eq!(image.dimensions(), (2, 2));
        assert_eq!(image.flatten_to_u8(), vec![pixels.to_vec()]);
        assert_eq!(image.depth(), before_depth);
        assert_eq!(image.metadata().is_premultiplied_alpha(), before_alpha);
        assert_eq!(image.metadata().color_trc(), Some(transfer));
    }
}

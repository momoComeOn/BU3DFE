function img_ = norm_img(img)
    minz = min(min(img));
    maxz = max(max(img));
    img_ = uint8(double(img - minz) ./ double(maxz - minz) * 255);
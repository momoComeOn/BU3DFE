function img_norm = get_img_norm(img)
    
    img = norm_img(img);
    
    [nx, ny, nz] = surfnorm(img);
    
    mask = uint8(img ~= 0);
    
    img_norm(:,:,1) = img;
    img_norm(:,:,2) = norm_img(nx) .* mask;
    img_norm(:,:,3) = norm_img(ny) .* mask;
    img_norm(:,:,4) = norm_img(nz) .* mask;
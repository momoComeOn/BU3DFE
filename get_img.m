function img = get_img(data)    
    cnt = size(data, 1);
    min_z = min(data(:,3));
    min_x = min(data(:,1));
    max_x = max(data(:,1));
    min_y = min(data(:,2));
    max_y = max(data(:,2));

    img = zeros(ceil(max_y) - floor(min_y) + 1, ceil(max_x) - floor(min_x) + 1);
    img(:,:) = min_z;
    for k=1:cnt
        ax = int32(data(k, 1) - min_x + 1);
        ay = int32(data(k, 2) - min_y + 1);
        img(ay, ax) = max(img(ay, ax), data(k, 3));
    end
    
    min_z = min(min(img));
    img = max(img, min_z);
    
    img = flipud(img);
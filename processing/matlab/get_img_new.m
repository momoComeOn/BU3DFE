function img = get_img_new(data)

    tri = delaunay(data(:,1), data(:,2));
    min_z = min(data(:,3));
    max_z = max(data(:,3));
    min_x = min(data(:,1));
    max_x = max(data(:,1));
    min_y = min(data(:,2));
    max_y = max(data(:,2));
    
    data(:,1) = data(:,1) - min_x;
    data(:,2) = data(:,2) - min_y;
    
    img = zeros(ceil(max_x) - floor(min_x) + 1, ceil(max_y) - floor(min_y) + 1);
    img(:,:) = min_z;
    
    p1 = data(tri(:,1),:);
    p2 = data(tri(:,2),:);
    p3 = data(tri(:,3),:);
    
    e1 = p1 - p2;
    e2 = p2 - p3;
    e3 = p3 - p1;
    
    min_zz = max_z;
    
    for k=1:length(tri)
        minx = ceil(min(min(data(tri(k,1),1), data(tri(k,2),1)), data(tri(k,3),1)));
        maxx = floor(max(max(data(tri(k,1),1), data(tri(k,2),1)), data(tri(k,3),1)));
        miny = ceil(min(min(data(tri(k,1),2), data(tri(k,2),2)), data(tri(k,3),2)));
        maxy = floor(max(max(data(tri(k,1),2), data(tri(k,2),2)), data(tri(k,3),2)));
        
        x = repmat([minx:maxx], 1, length([miny:maxy]));
        y = repmat([miny:maxy], length([minx:maxx]), 1);
        y = reshape(y, 1, size(y, 1)*size(y, 2));
        point = [x;y]';
        
        flag1 = (point(:,1) - p2(k,1))*e1(k,2) - (point(:,2) - p2(k,2))*e1(k,1) > 0;
        flag2 = (point(:,1) - p3(k,1))*e2(k,2) - (point(:,2) - p3(k,2))*e2(k,1) > 0;
        flag3 = (point(:,1) - p1(k,1))*e3(k,2) - (point(:,2) - p1(k,2))*e3(k,1) > 0;
        
        n = cross(e1(k,:), e2(k,:));
        c = dot(p1(k,:), n);
        
        idx = find((flag1 == flag2) & (flag2 == flag3) == 1);
        if (length(idx) == 0)
            continue;
        end
        
        for i=1:length(idx)
            img(point(idx(i),1), point(idx(i),2)) = (c - dot(n(1:2), point(idx(i),:))) / n(3);
            min_zz = min(min_zz, img(point(idx(i),1), point(idx(i),2)));
        end
    end
    img = max(img, min_zz);
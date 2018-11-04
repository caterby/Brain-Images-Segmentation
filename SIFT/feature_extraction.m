function feature_pixel =feature_extraction(I)
    % 1. Detect corners using Harris–Stephens algorithm
    corners = detectHarrisFeatures(I);
    [features_1, valid_corners_1] = extractFeatures(I, corners);
    %disp(size(valid_corners.Location));
    figure; imshow(I); hold on;
    plot(valid_corners_1);

    % 2. Detect SURF features
    points = detectSURFFeatures(I);
    [features_2, valid_points_2] = extractFeatures(I, points);
%     figure; imshow(I); hold on;
%     plot(valid_points_2.selectStrongest(10),'showOrientation',true);

    % 3. Detect MSER features
    regions = detectMSERFeatures(I);
    [features_3, valid_points_3] = extractFeatures(I,regions,'Upright',true);
%     figure; imshow(I); hold on;
%     plot(valid_points_3,'showOrientation',true);

    % 4. Detect BRISK features(corners)
    points = detectBRISKFeatures(I);
    [features_4, valid_corners_4] = extractFeatures(I, points);
    %disp(size(valid_corners.Location));
%     figure; imshow(I); hold on;
%     plot(valid_corners_4);

    % 5. Detect corners using FAST algorithm(corners)
    corners = detectFASTFeatures(I);
    [features_5, valid_corners_5] = extractFeatures(I, corners);
    %disp(size(valid_corners.Location));
%     figure; imshow(I); hold on;
%     plot(valid_corners_5);

    % 6. Detect corners using minimum eigenvalue algorithm
    corners = detectMinEigenFeatures(I);
    [features_6, valid_corners_6] = extractFeatures(I, corners);
    %disp(size(valid_corners.Location));
%     figure; imshow(I); hold on;
%     plot(valid_corners_6);

    % 7. Extract local binary pattern (LBP) features
    %features_7 = extractLBPFeatures(I);

    % 8. Extract histogram of oriented gradients (HOG) features
    strongest = selectStrongest(valid_corners_5,10);
    [features_8, valid_points_8, ptVis] = extractHOGFeatures(I,strongest);
    %[features_8,hogVisualization] = extractHOGFeatures(I);
    %[features,validPoints] = extractHOGFeatures(I,points)

%     figure;
%     imshow(I);
%     hold on;
%     plot(ptVis);

    % Construct feature for each pixel(simplest, binary): 
    % [Harris–Stephens, SURF features, MSER features, BRISK features, FAST algorithm, 
    %  minimum eigenvalue algorithm, LBP features, HOG features]
    feature_pixel = zeros(240,240,7);

    %1 Harris–Stephens
    points_location_1 = round(valid_corners_1.Location);
    [m,~] = size(points_location_1);
    for i = 1:m
         feature_pixel(points_location_1(i,1), points_location_1(i,2), 1) = 1;
    end
    %disp(sum(sum(feature_pixel(:,:,1))));

    %2 SURF features
    points_location_2 = round(valid_points_2.Location);
    [m,~] = size(points_location_2);
    for i = 1:m
         feature_pixel(points_location_2(i,1), points_location_2(i,2), 2) = 1;
    end

    %3 MSER features
    points_location_3 = round(valid_points_3.Location);
    [m,~] = size(points_location_3);
    for i = 1:m
         feature_pixel(points_location_3(i,1), points_location_3(i,2), 3) = 1;
    end

    %4 BRISK features
    points_location_4 = round(valid_corners_4.Location);
    [m,~] = size(points_location_4);
    for i = 1:m
         feature_pixel(points_location_4(i,1), points_location_4(i,2), 4) = 1;
    end

    %5 corners using FAST algorithm(corners)
    points_location_5 = round(valid_corners_5.Location);
    [m,~] = size(points_location_5);
    for i = 1:m
         feature_pixel(points_location_5(i,1), points_location_5(i,2), 5) = 1;
    end

    %6 corners using minimum eigenvalue algorithm
    points_location_6 = round(valid_corners_6.Location);
    [m,~] = size(points_location_6);
    for i = 1:m
         feature_pixel(points_location_6(i,1), points_location_6(i,2), 6) = 1;
    end

    %8 histogram of oriented gradients (HOG) features
    points_location_8 = round(valid_points_8.Location);
    [m,~] = size(points_location_8);
    for i = 1:m
         feature_pixel(points_location_8(i,1), points_location_8(i,2), 7) = 1;
    end
end
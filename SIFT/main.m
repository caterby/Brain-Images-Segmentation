
% BRATS2015_Training\HGG\
root_dir = 'BRATS2015_Training\HGG\';
root_dir_files = dir(root_dir);
% Consider file '.' and '..'
num_patients = length(root_dir_files) - 2;

% One naive bayes classifier, 240*240*16 samples, each 7 dimensions
% Initialize feature vector
samples_x = zeros(num_patients,240,240,10,6);
samples_y = zeros(num_patients,240,240,10);

count = 0;

for root_file = root_dir_files'
    
    
    tic;
    
    disp(root_file.name);
    
    if strcmp(root_file.name, '.') || strcmp(root_file.name, '..')
        continue; 
    end
    
%     if count == 6
%         break;
%     end
    count = count + 1;
    
    
    % BRATS2015_Training\HGG\brats_2013_pat0001_1
    dir2 = strcat(root_dir, root_file.name);
    dir2_files = dir(dir2);

    % BRATS2015_Training\HGG\brats_2013_pat0001_1\VSD.Brain.XX.O.MR_Flair.54512
    dir3 = strcat(dir2, '\', dir2_files(3).name);
    dir3_files = dir(dir3);    
    x_data_file = strcat(dir3, '\', dir3_files(end).name);

    info = mha_read_header(x_data_file); 
    V = mha_read_volume(info);
    for i = 1:10
        one_img = uint16(squeeze(V(:,:,(i-1)*10 + 31)));%change the value of round can get different img with differen view, eg. round(1~155)
        I = mat2gray(one_img);
        %imshow(I);
        
        % Feature 1: pixel gray value
        for j = 1:240
            for k = 1:240
                samples_x(count,j,k,i,1) = round(I(j,k) * 10);
            end
        end

        % SIFT feature point extraction
        [kpl, kpmag] = sift(I);
        sift_point = [0 0];
        
        for j = 1:length(kpmag)
            if kpmag(j) == min(kpmag)
                sift_point(1) = round(kpl(j*2));
                sift_point(2) = round(kpl(j*2-1));
                break;
            end
        end
        
        
        % Feature 2: hamming distance from SIFT tumor point(48 intervals)
        for j = 1:240
            for k = 1:240
                distance = abs(sift_point(1) - j) + abs(sift_point(2) - k);
                samples_x(count,j,k,i,2) = round(distance / 48);
            end
        end
        
        % Feature 3: pixel value different with SIFT tumor point
        pixel_value_sift = I(sift_point(1),sift_point(2));
        for j = 1:240
            for k = 1:240
                samples_x(count,j,k,i,3) = abs(round((I(j,k) - pixel_value_sift) * 10));
            end
        end
        
%         % Feature 4: whether within certain region(SIFT tumor point + corners)
%         corners = detectHarrisFeatures(I);
%         [~, valid_corners] = extractFeatures(I, corners);
%         points_location = round(valid_corners.Location);
%         [num_corners, ~] = size(points_location);
%         
%         for j = 1:240
%             for k = 1:240
%                 samples_x(count,j,k,i,4) = j;
%             end
%         end
        
        % Feature 5: x value
        for j = 1:240
            for k = 1:240
                samples_x(count,j,k,i,4) = j;
            end
        end
        
        % Feature 6: y value
        for j = 1:240
            for k = 1:240
                samples_x(count,j,k,i,5) = k;
            end
        end
        
        % Feature 7: z value
        for j = 1:240
            for k = 1:240
                samples_x(count,j,k,i,6) = i;
            end
        end
        
    end
    

    % BRATS2015_Training\HGG\brats_2013_pat0001_1\VSD.Brain_3more.XX.O.OT.54517
    dir4 = strcat(dir2, '\', dir2_files(end).name);
    dir4_files = dir(dir4);
    y_data_file = strcat(dir4, '\', dir4_files(end).name);

    info = mha_read_header(y_data_file); 
    V = mha_read_volume(info);
    for i = 1:10
        one_img = uint16(squeeze(V(:,:,(i-1)*10 + 31)));%change the value of round can get different img with differen view, eg. round(1~155)
        %I = mat2gray(one_img);
        %imshow(I);
        samples_y(count,:,:,i) = one_img;
    end
    
    toc;
    
end

% Naive Bayes training
X_train = reshape(samples_x(1:round(8*count/10),:,:,:,:), [round(8*count/10)*240*240*10 6]);
Y_train = reshape(samples_y(1:round(8*count/10),:,:,:), [round(8*count/10)*240*240*10 1]);
%Y_train(:,1) = [];
Mdl = fitcnb(X_train,Y_train);

% Naive Bayes testing
x_test = reshape(samples_x(round(8*count/10)+1:count,:,:,:,:), [(count-round(8*count/10))*240*240*10 6]);
y_real = reshape(samples_y(round(8*count/10)+1:count,:,:,:), [(count-round(8*count/10))*240*240*10 1]);
y_test = predict(Mdl,x_test);

% Print result(including normal points)
result = sum(y_test == y_real);
disp('Correctly classified points(including normal points):')
disp(result);
disp('Accuracy(including normal points):')
disp(result/length(y_real));

% Print result(not including normal points)
result = 0;
count = 0;
for i = 1:length(y_real)
    if y_real(i) == 0
        continue;
    end
    count = count + 1;
    if y_real(i) == y_test(i)
        result = result + 1; 
    end
end
disp('Correctly classified points(not including normal points):')
disp(result);
disp('Accuracy(not including normal points):')
disp(result/count);





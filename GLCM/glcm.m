%Greatest Credit to Manu BN
%https://www.mathworks.com/matlabcentral/fileexchange/55107-brain-mri-tumor-detection-and-classification
function [] = glcm()
    tic 
    data_mine('C:\Users\mark\Downloads\BRATS2015_Training\', 'LGG'); %change the dataset path if needed
    toc
    data_mine('C:\Users\mark\Downloads\BRATS2015_Training\', 'HGG');
    toc
    data_mine('C:\Users\mark\Downloads\', 'BRATS2015_Training');
    toc
  
    function F = extract(I)
        %gray = rgb2gray(I);
        % Otsu Binarization for segmentation
        %level = graythresh(I);
        
        %gray = gray>80;
        img = im2bw(I, .6);
        img = bwareaopen(img, 80); 
        img2 = im2bw(I);
        
        % Try morphological operations
        %gray = rgb2gray(I);
        %tumor = imopen(gray,strel('line',15,0));
        
        %subplot(1, 2, 1);
        %imshow(I);
        %title('Brain MRI Image');
        %subplot(1, 2, 2);
        %imshow(img);
        %title('Segmented Image');

        signal1 = img2(:, :);
        %Feat = getmswpfeat(signal,winsize,wininc,J,'matlab');
        %Features = getmswpfeat(signal,winsize,wininc,J,'matlab');

        [cA1, cH1, cV1, cD1] = dwt2(signal1, 'db4');
        [cA2, cH2, cV2, cD2] = dwt2(cA1, 'db4');
        [cA3, cH3, cV3, cD3] = dwt2(cA2, 'db4');

        DWT_feat = [cA3, cH3, cV3, cD3];
        G = pca(DWT_feat);
        %whos DWT_feat
        %whos G
        g = graycomatrix(G);
        stats = graycoprops(g, 'Contrast Correlation Energy Homogeneity');
        Contrast = stats.Contrast;
        Correlation = stats.Correlation;
        Energy = stats.Energy;
        Homogeneity = stats.Homogeneity;
        Mean = mean2(G);
        Standard_Deviation = std2(G);
        Entropy = entropy(G);
        RMS = mean2(rms(G));
        %Skewness = skewness(img)
        Variance = mean2(var(double(G)));
        a = sum(double(G(:)));
        Smoothness = 1 - (1 / (1 + a));
        Kurtosis = kurtosis(double(G(:)));
        Skewness = skewness(double(G(:)));
        % Inverse Difference Movement
        m = size(G, 1);
        n = size(G, 2);
        in_diff = 0;
        for i = 1 : m
            for j = 1 : n
                temp = G(i, j)./(1 + (i - j).^2);
                in_diff = in_diff+temp;
            end
        end
        IDM = double(in_diff);

        %F = [Contrast, Correlation, Energy, Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];
        F = [Contrast, Correlation, Energy, Homogeneity, Mean, Standard_Deviation, Entropy, Variance, Smoothness, Kurtosis, Skewness, IDM];
        %whos F;
        %display(F);
    end

    function [] = data_mine(base_path, LGG_HGG)
        fl_mha_fp = strcat(base_path, LGG_HGG, '\**\brats_*\VSD.Brain.XX.O.MR_Flair.*\VSD.Brain.XX.O.MR_Flair.*.mha');
        fl_mha_files = dir(fl_mha_fp);
        %target_slices = 71 : 86;
        %target_slices = [31, 41, 51, 61, 71, 81, 91, 101, 111, 121];
        target_slices = 1 : 155;
        slice_num = size(target_slices, 2);
        
        meas = zeros(length(fl_mha_files) * slice_num, 12);
        label = zeros(length(fl_mha_files) * slice_num, 1);
        
        rng(1000);
        permutation_array = randperm(length(fl_mha_files));
        
        for k = 1 : length(fl_mha_files)
            i = permutation_array(k);
        
            fl_mha_info = mha_read_header(strcat(fl_mha_files(i).folder, '\', fl_mha_files(i).name));
            fl_mha_vectors = mha_read_volume(fl_mha_info);
            
            ot_mha_files = dir(strcat(fl_mha_files(i).folder, '\..\VSD.Brain*OT*.*\VSD.Brain*OT*.mha'));
            ot_mha_info = mha_read_header(strcat(ot_mha_files(1).folder, '\', ot_mha_files(1).name));
            ot_mha_vectors = mha_read_volume(ot_mha_info);
            fprintf('[%d => %d]\n', k, length(fl_mha_files));
            
            for j = 1 : slice_num
                slide_id = target_slices(1, j);
                fl_img = mat2gray(uint16(squeeze(fl_mha_vectors(:, :, slide_id))));
                ot_mat = uint16(squeeze(ot_mha_vectors(:, :, slide_id)));
                %subplot(1, 2, 1);
                %imshow(fl_img);
                %title('MRI T2 Image');
                %subplot(1, 2, 2);
                %imshow(mat2gray(ot_mat));
                %title('MRI OT Image');
                feat_id = (i - 1)* slice_num + j;
                meas(feat_id, :) = extract(fl_img);
                
                x = hist(ot_mat(:), 0 : 4);
                x(1) = 0;
                [~, y] = max(x);
                label(feat_id, 1) = y - 1;
            end
        end
        
        train_size = round(size(meas, 1) * 0.8);
        O1 = fitNaiveBayes(meas(1 : train_size, :), label(1 : train_size));
        species = O1.predict(meas(train_size + 1: end, :));
        result = label(train_size + 1: end) - species;
        correct = sum(result(:) == 0);
        accuracy = correct * 100.0 / length(result);
        fprintf('%s accuracy: %f%%.\n', LGG_HGG, accuracy);
        save(strcat(LGG_HGG, '.mat'), 'meas', 'label', 'accuracy');
    end
end
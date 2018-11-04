%training
files = dir('BRATS2015_Training/HGG');
count_pixel = 1;
train = [];
train_class = [];
for i = 1: length(files)
    if strfind(files(i).name, 'brats')
        
        dir2 = strcat('BRATS2015_Training/HGG/', files(i).name);
        files2 = dir(dir2);
        for j = 1: length(files2)
            if strfind(files2(j).name, 'VSD')
                dir3 = strcat(dir2, '/');
                dir4 = strcat(dir3, files2(j).name);
                dir5 = strcat(dir4, '/*.mha');
                file3 = dir(dir5);
                dir6 = strcat(dir4, '/');
                dir7 = strcat(dir6, file3.name);
                disp(dir7);
                info = mha_read_header(dir7); 
                V = mha_read_volume(info);
                if strfind(file3.name, 'OT')
                    my = [];
                    cnt = 1;
                    for p = 1: 10: 155
                        my(:, :, cnt) = uint16(squeeze(V(:,:,round(p))));%change the value of round can get different img with differen view, eg. round(1~155)
                        cnt = cnt + 1;
                    end
                elseif strfind(file3.name, 'Flair')
                    my2 = [];
                    cnt = 1;
                    I2 = [];
                    for p = 1: 10: 155
                        my2(:, :, cnt) = uint16(squeeze(V(:,:,round(p))));
                        I2(:, :, cnt) = mat2gray(my2(:, :, cnt));
                        cnt = cnt + 1;
                    end
                %my2 = im2uint8(my)
                %disp(my);
                       
                end
            end
        end
        %I2 = mat2gray(my2);
        for cnt = 1: 16
            for w = 1: 10: 240
                for h = 1: 10: 240
                    %if my(w, h) == 0
        %if my(i, j) == 4
                        train(count_pixel, 1) = calEnergy(I2(:, :, cnt), w, h);
                    %disp(score(count_pixel, 1));
                        train(count_pixel, 2) = calContrast(I2(:, :, cnt), w, h);
               
                        train_class(count_pixel, 1) = my(w, h, cnt);
                        count_pixel = count_pixel + 1;
                    %end
                
                end
            end
        end  
        
    end
end


%naive bayes
nb = fitcnb(train, train_class);

%testing
files = dir('BRATS2015_Testing/HGG');
count_pixel = 1;
test = [];
test_class = [];
for i = 1: length(files)
    if strfind(files(i).name, 'brats')
        
        dir2 = strcat('BRATS2015_Testing/HGG/', files(i).name);
        files2 = dir(dir2);
        for j = 1: length(files2)
            if strfind(files2(j).name, 'VSD')
                dir3 = strcat(dir2, '/');
                dir4 = strcat(dir3, files2(j).name);
                dir5 = strcat(dir4, '/*.mha');
                file3 = dir(dir5);
                dir6 = strcat(dir4, '/');
                dir7 = strcat(dir6, file3.name);
                disp(dir7);
                info = mha_read_header(dir7); 
                V = mha_read_volume(info);
                if strfind(file3.name, 'OT')
                    my = [];
                    cnt = 1;
                    for p = 1: 10: 155
                        my(:, :, cnt) = uint16(squeeze(V(:,:,round(p))));%change the value of round can get different img with differen view, eg. round(1~155)
                        cnt = cnt + 1;
                    end
                elseif strfind(file3.name, 'Flair')
                    my2 = [];
                    cnt = 1;
                    I2 = [];
                    for p = 1: 10: 155
                        my2(:, :, cnt) = uint16(squeeze(V(:,:,round(p))));
                        I2(:, :, cnt) = mat2gray(my2(:, :, cnt));
                        cnt = cnt + 1;
                    end
                %my2 = im2uint8(my)
                %disp(my);
                       
                end
            end
        end
        %I2 = mat2gray(my2);
        for cnt = 1: 16
            for w = 1: 10: 240
                for h = 1: 10: 240
                    %if my(w, h, cnt) ~= 0%only calculate 1-4 labels' accuracy
                        %disp(my(w, h, cnt));
        %if my(i, j) == 4
                        test(count_pixel, 1) = calEnergy(I2(:, :, cnt), w, h);
                    %disp(score(count_pixel, 1));
                        test(count_pixel, 2) = calContrast(I2(:, :, cnt), w, h);
               
                        test_class(count_pixel, 1) = my(w, h, cnt);
                        count_pixel = count_pixel + 1;
                    %end
                
                end
            end
        end  
        
    end
end

predict_label = predict(nb, test);
accuracy = length(find(predict_label == test_class))/length(test_class)*100;
disp(accuracy);

function score = calEnergy(image, y, x)
score = 0;
if y+3 <= 240
    score = score + (image(y+3,x+0))^2;
end
if y+3 <= 240 && x+1 <= 240
    score = score + (image(y+3,x+1))^2;
end
if y+2 <=240 && x+2 <=240    
    score = score + (image(y+2,x+2))^2;
end
if y+1 <= 240 && x+3 <= 240    
    score = score + (image(y+1,x+3))^2;
end
if x+3 <= 240     
    score = score + (image(y+0,x+3))^2;
end
if y-1 >= 1 && x+3 <= 240    
    score = score + (image(y-1,x+3))^2;
end
if y-2 >=1 && x+2 <=240    
    score = score + (image(y-2,x+2))^2;
end
if y-3 >=1 && x+1 <=240    
    score = score + (image(y-3,x+1))^2; 
end
if y-3 >=1    
    score = score + (image(y-3,x+0))^2; 
end
if y+3 <=240 && x-1 >=1    
    score = score + (image(y+3,x-1))^2;
end
end

function score = calContrast(image, y, x)
score = 0;
if y+3 <= 240
    score = score + abs(image(y, x) - image(y+3,x+0));
end
if y+3 <= 240 && x+1 <= 240
    score = score + abs(image(y, x) - image(y+3,x+1));
end
if y+2 <=240 && x+2 <=240    
    score = score + abs(image(y, x) - image(y+2,x+2));
end
if y+1 <= 240 && x+3 <= 240    
    score = score + abs(image(y, x) - image(y+1,x+3));
end
if x+3 <= 240     
    score = score + abs(image(y, x) - image(y+0,x+3));
end
if y-1 >= 1 && x+3 <= 240    
    score = score + abs(image(y, x) - image(y-1,x+3));
end
if y-2 >=1 && x+2 <=240    
    score = score + abs(image(y, x) - image(y-2,x+2));
end
if y-3 >=1 && x+1 <=240    
    score = score + abs(image(y, x) - image(y-3,x+1)); 
end
if y-3 >=1    
    score = score + abs(image(y, x) - image(y-3,x+0)); 
end
if y+3 <=240 && x-1 >=1    
    score = score + abs(image(y, x) - image(y+3,x-1));
end
end
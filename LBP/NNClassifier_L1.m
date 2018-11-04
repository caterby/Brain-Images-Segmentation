
function [final_accu,PreLabel] = NNClassifier_L1(Samples_Train,Samples_Test,Labels_Train,Labels_Test)

Train_Model = Samples_Train;
Test_Model = Samples_Test;
numTest = size(Test_Model,2);
numTrain = size(Train_Model,2);

PreLabel = [];

for test_sample_no = 1:numTest
   
    testMat = repmat(Test_Model(:,test_sample_no), 1, numTrain);
    scores_vec = cal_matrix_distance(testMat, Train_Model);

    [min_val min_idx] = min(scores_vec);
    best_label = Labels_Train(1,min_idx);
    PreLabel = [PreLabel, best_label];
end



Comp_Label = PreLabel - Labels_Test;
final_accu = (sum((Comp_Label==0))/numel(Comp_Label))*100

end

function disVec=cal_matrix_distance(mat1,mat2)
%using L1 as the distance metric
disVec = sum(abs(mat1 - mat2), 1);
%you may add other distance matric here:
%....

end
        
                
            
            
            
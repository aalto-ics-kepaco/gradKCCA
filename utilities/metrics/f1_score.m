function [f1] = f1_score(out,cor)

out = abs(out);
out(out < 0.05) = 0;

 TP = 0; TN = 0; FP = 0; FN = 0;
for i = 1:size(out,1)
    if cor(i) == 1 && out(i) > 0
        TP = TP + 1;
        
    elseif cor(i) == 0 && out(i) > 0
        FP = FP + 1;
        
    elseif cor(i) == 0 && out(i) == 0
        TN = TN + 1;
        
    elseif cor(i) == 1 && out(i) == 0
        FN = FN + 1;
    end
end


f1 = 2 * TP / ( 2 * TP + FN + FP );
    
    
end


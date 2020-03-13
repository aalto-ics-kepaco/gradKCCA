function [val_auc,fpr,tpr] = calc_auc(pred,labels)

labels = abs(labels);
pred = abs(pred);
pred(pred > 0.05) = 1;

nb_pos = sum(labels == 1);
nb_neg = sum(labels == 0);
nb_tot = nb_pos + nb_neg;
[predf,idx] = sort(abs(pred),'descend');
labelsf = labels(idx);
tp = cumsum(labelsf);
fp = ((1:nb_tot) - tp);
flags = (diff(predf) ~=0);
tpr = tp(flags) / nb_pos;
fpr = fp(flags) / nb_neg;
tpr = [0 tpr 1];
fpr = [0 fpr 1];
val_auc = sum((fpr(2:end)-fpr(1:end-1)).* (tpr(2:end)+tpr(1:end-1)))/2;

end
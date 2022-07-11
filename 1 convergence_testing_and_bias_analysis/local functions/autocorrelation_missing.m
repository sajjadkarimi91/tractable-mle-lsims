function Cxy = autocorrelation_missing( X , Y , Lags)

%Cxy(T)=E{X(t)Y(t-T)}

% x_temp = X(:);
% y_temp = Y(:);
% 
% x_temp(isnan(x_temp))=0;
% y_temp(isnan(y_temp))=0;
% 
% [x_temp,temp_index] =sort(x_temp,'descend');
% X(temp_index(1:ceil(0.03*length(X))))=nan;
% 
% [y_temp,temp_index] =sort(y_temp,'descend');
% Y(temp_index(1:ceil(0.03*length(Y))))=nan;

Cxy = zeros(1,2*Lags+1);
Cxy = Cxy *nan;
Cxy(Lags+1)=1;

for k = 1:Lags
    
    temp1 = X(1:end-k);
    temp2= Y(k+1:end);
    
    ind_nan1 = isnan(temp1);
    ind_nan2 = isnan(temp2);    
    indnan = ind_nan1|ind_nan2;
    
    temp1 = temp1(~indnan);
    temp2 = temp2(~indnan);
    
    if(length(temp1)>2)        
        Cxy(Lags+1-k) = (temp1(:)'-mean(temp1(:)))*(temp2(:)-mean(temp2(:)))/sqrt((length(temp1)-1)^2*var(temp1)*var(temp2));
    end
    
end


for k = 0:Lags
    
    temp1 = X(k+1:end);
    temp2= Y(1:end-k);
    
    ind_nan1 = isnan(temp1);
    ind_nan2 = isnan(temp2);    
    indnan = ind_nan1|ind_nan2;
    
    temp1 = temp1(~indnan);
    temp2 = temp2(~indnan);
    
    if(length(temp1)>2)        
        Cxy(k+1+Lags) =  (temp1(:)'-mean(temp1(:)))*(temp2(:)-mean(temp2(:)))/sqrt(length(temp1)^2*var(temp1)*var(temp2));
    end
    
end


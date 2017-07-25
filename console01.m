
close all; clear;
d = 2;
k = 3;
n = 1000;
[X,label] = mixGaussRnd(d,k,n);
plotClass(X,label);

m = floor(n/2);
X1 = X(:,1:m);
X2 = X(:,(m+1):end);
% train
[z1,model,llh] = mixGaussEm(X1,k);

% gmm = cgmmInit(X1',k,1000,1);
% gmm=cgmmProcess(gmm);


gmm2 = cgmmInit(X1',k,5000,1);
gmm2 = cgmmProcessOnline(gmm2);

fprintf('mixGuassEm model.w:\n');
model.w
fprintf('gmm gmm.Guass.Alpha:\n');
gmm.Guass.Alpha
fprintf('mixGuassEm model.mu:\n');
model.mu
fprintf('mixGuassEm gmm.Guass.Mean:\n');
gmm.Guass.Mean
fprintf('mixGuassEm model.Sigma:\n');
model.Sigma
fprintf('mixGuassEm gmm.Guass.Sigma:\n');
gmm.Guass.Sigma



function gmm=cgmmInit(dataSet,NB_Guass,maxIter,onlineMaxIter,Mean,Alpha,Sigma)
	gmm.dataSet = dataSet;
	gmm.NB_Guass = NB_Guass;
	gmm.maxIter = maxIter;
    gmm.onlineMaxIter = onlineMaxIter;
	% R is   6 * 6 * NB_Guass;
	% fai is  len(dataSet) * NB_Guass 
    if nargin == 7
        gmm.Guass.Sigma  = Sigma;
        gmm.Guass.Mean = Mean;
        gmm.Guass.Alpha = Alpha;
    else
        l = size(gmm.dataSet);
        temp  = diag(ones(l(2),1));
        gmm.Guass.Sigma = zeros(l(2),l(2),gmm.NB_Guass);
        for index = 1:gmm.NB_Guass
            gmm.Guass.Sigma(:,:,index) = temp;
        end
        gmm.Guass.Mean = ones(l(2),1)*(1:gmm.NB_Guass);
        gmm.Guass.Alpha = 1/gmm.NB_Guass*ones(gmm.NB_Guass,1);
    end   
end

function gmm=cgmmProcess(gmm)
	[len,~] = size(gmm.dataSet);
    label = ceil(gmm.NB_Guass*rand(1,len));
    
    logP = zeros(len,gmm.NB_Guass);
    P = full(sparse(1:len,label,1,len,gmm.NB_Guass,len));
	for j=1:gmm.maxIter
        
    %% M step:
		gmm.Guass.Mean = bsxfun(@times,gmm.dataSet'*P,1./sum(P));
		for i = 1:gmm.NB_Guass
            %gmm.Guass.R(:,:,i) =0;
            %gmm.Guass.Sigma(:,:,i) = bsxfun(@times,conj(gmm.dataSet-gmm.Guass.Mean(:,i)),sqrt(P(:,i)));
            Xo = bsxfun(@minus,gmm.dataSet,gmm.Guass.Mean(:,i)');
            Xo = bsxfun(@times,Xo,sqrt(P(:,i)));
            gmm.Guass.Sigma(:,:,i) = conj(Xo')*Xo/sum(P(:,i));
            gmm.Guass.Alpha(i) = 1/len*sum(P(:,i));            
        end
        
		%% E step:
		for Index=1:len
			for i=1:gmm.NB_Guass
				logP(Index,i) = log(gmm.Guass.Alpha(i))+logProbcalcGuass(gmm.dataSet(Index,:)', gmm.Guass.Mean(:,i),...
					gmm.Guass.Sigma(:,:,i));
			end
			logP(Index,:) = logP(Index,:) - max(logP(Index,:));
			P(Index,:)=exp(logP(Index,:))/sum(exp(logP(Index,:)));				
		end

    end
end

function gmm = cgmmProcessOnline(gmm)
    [len,~] = size(gmm.dataSet);
    label = ceil(gmm.NB_Guass*rand(1,len));

    logP = [0,0,1];
    %     P = full(sparse(1:len,label,1,len,gmm.NB_Guass,len));
    % Main loop of the online EM algorithm
     MeanPre = gmm.Guass.Mean ;
     SigmaPre = gmm.Guass.Sigma ;
     AlphaPre =  gmm.Guass.Alpha;
    for Index2  = 1:gmm.onlineMaxIter
    	for Index = 1:len
    		[Mean,Sigma,Alpha] = gm_online_step(gmm.dataSet(Index,:),MeanPre,SigmaPre,AlphaPre,1/Index);
    		MeanPre = Mean;
    		SigmaPre = Sigma;
    		AlphaPre = Alpha;
    	end
    end
    gmm.Guass.Mean = Mean;
    gmm.Guass.Alpha = Alpha;
    gmm.Guass.Sigma = Sigma;

end
function [Mean,Sigma,Alpha] = gm_online_step(data,MeanPre,SigmaPre,AlphaPre,gamman)
    NB_Guass = length(AlphaPre);
    logPro = zeros(NB_Guass,1);
    for i =1:NB_Guass
        logPro(i)=logProbcalcGuass(data',MeanPre(:,i),SigmaPre(:,:,i));
    end
	logpostProb = (log(AlphaPre)+logPro)-max((log(AlphaPre)+logPro));
	postProb = exp(logpostProb)/sum(exp(logpostProb));	
    Alpha = (1-gamman)*AlphaPre+gamman*postProb;
	Mean = (1-gamman)*MeanPre+gamman*data'*postProb';
    for i = 1:NB_Guass
        Sigma(:,:,i) = (1-gamman)*SigmaPre(:,:,i)+gamman*postProb(i)*(data'-Mean(:,i))*(data'-Mean(:,i))';
    end

end
function logPro=logProbcalcGuass(x,meane,varR)
	n=length(x);
%	Pro = 1/((2*pi)^(n/2)*abs(det(varR)))*exp(-1/2*(x-meane)*inv(varR)*conj((x-meane)'));
	varRDet = abs(det(varR));
	varRInv = inv(varR);
	coeff = -(n/2)*log(2*pi)-1/2*log(varRDet);
	retr = -1/2*conj((x-meane)')*varRInv*(x-meane);
	logPro = coeff + retr;
end
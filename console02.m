% refrence:
% online learning for audio cluster and segmentation
clear;
% close all;
MU1 = [3 0];
SIGMA1 = [2 1; 1 4];
MU2 = [-2 3];
SIGMA2 = [3 -1; -1 2];
ALPHA = 0.5;
N = 4000;
k = 2;
X = GaussMixRnd(MU1,SIGMA1,MU2,SIGMA2,ALPHA,N);

%% give the traindata in two step;
%     r1=mvnrnd(MU1,SIGMA1,N);
%     r2=mvnrnd(MU2,SIGMA2,N);
%     [X] = [r1;r2];
% % train
[z1,model,llh] = mixGaussEm(X',k);
% gmm = cgmmInit(X,2,1000,1);
% gmm=cgmmProcess(gmm);
muinit = [1,-1;0,0];
sigmainit = zeros(2,2,2);
sigmainit(:,:,1) = eye(2,2);
sigmainit(:,:,2) = eye(2,2);
gmm2 = cgmmInit(X,k,5000,1,[0.5,0.5]',muinit,sigmainit);
[gmm2,Mu1,Mu2] = cgmmProcessOnline(gmm2);
figure(1);
plot(Mu1(:,1));hold on;plot(Mu1(:,2));
figure(2);
plot(Mu2(:,1));hold on;plot(Mu2(:,2));


function gmm=cgmmInit(dataSet,NB_Guass,maxIter,onlineMaxIter,Alpha,Mean,Sigma)
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

% function gmm = cgmmProcessOnline(gmm)
%     [len,~] = size(gmm.dataSet);
%     label = ceil(gmm.NB_Guass*rand(1,len));
% 
%     logP = [0,0,1];
%     %     P = full(sparse(1:len,label,1,len,gmm.NB_Guass,len));
%     % Main loop of the online EM algorithm
%      MeanPre = gmm.Guass.Mean ;
%      SigmaPre = gmm.Guass.Sigma ;
%      AlphaPre =  gmm.Guass.Alpha;
%     for Index2  = 1:gmm.onlineMaxIter
%     	for Index = 1:len
%     		[Mean,Sigma,Alpha] = gm_online_step(gmm.dataSet(Index,:),MeanPre,SigmaPre,AlphaPre,1/Index);
%     		MeanPre = Mean;
%     		SigmaPre = Sigma;
%     		AlphaPre = Alpha;
%     	end
%     end
%     gmm.Guass.Mean = Mean;
%     gmm.Guass.Alpha = Alpha;
%     gmm.Guass.Sigma = Sigma;
% 
% end
% function [Mean,Sigma,Alpha] = gm_online_step(data,MeanPre,SigmaPre,AlphaPre,gamman)
%     NB_Guass = length(AlphaPre);
%     logPro = zeros(NB_Guass,1);
%     for i =1:NB_Guass
%         logPro(i)=logProbcalcGuass(data',MeanPre(:,i),SigmaPre(:,:,i));
%     end
% 	logpostProb = (log(AlphaPre)+logPro)-max((log(AlphaPre)+logPro));
% 	postProb = exp(logpostProb)/sum(exp(logpostProb));	
%     Alpha = (1-gamman)*AlphaPre+gamman*postProb;
% 	Mean = (1-gamman)*MeanPre+gamman*data'*postProb';
%     for i = 1:NB_Guass
%         Sigma(:,:,i) = (1-gamman)*SigmaPre(:,:,i)+gamman*postProb(i)*(data'-Mean(:,i))*(data'-Mean(:,i))';
%     end
% end
function logPro=logProbcalcGuass(x,meane,varR)
	n=length(x);
%	Pro = 1/((2*pi)^(n/2)*abs(det(varR)))*exp(-1/2*(x-meane)*inv(varR)*conj((x-meane)'));
	varRDet = abs(det(varR));
	varRInv = inv(varR);
	coeff = -(n/2)*log(2*pi)-1/2*log(varRDet);
	retr = -1/2*conj((x-meane)')*varRInv*(x-meane);
	logPro = coeff + retr;
end




function [S_0,S_1,S_2] = EstepInit(K,len)
    S_0 = zeros(K,1);
    S_1 = zeros(len,K);
    S_2 = zeros(2,2,2);
%     S_2(:,:,1) = eye(2,2);
%     S_2(:,:,2) = eye(2,2);
end

function [S_0,S_1,S_2] = Estep(data,Guass,S_0,S_1,S_2,Gamma)
    K = length(S_0);
    logP = zeros(K,1);
    for i = 1:K
        logP(i) = log(Guass.Alpha(i)) + logProbcalcGuass(data,Guass.Mean(:,i),...
            Guass.Sigma(:,:,i));
    end
    logP = logP - max(logP);
    P = exp(logP)/sum(exp(logP));
    S_0 =  Gamma*P + (1-Gamma)*S_0;
    S_1 = Gamma*bsxfun(@times,P',data) + (1-Gamma)*S_1;
    for i =1:K
        S_2(:,:,i) = Gamma*P(i)*data*data'+(1-Gamma)*S_2(:,:,i);
    end
end

function [gmm] = Mstep(gmm,S_0,S_1,S_2)
    gmm.Guass.Alpha = S_0;
    gmm.Guass.Mean = bsxfun(@rdivide,S_1,S_0');
    gmm.Guass.Sigma(:,:,1) = bsxfun(@rdivide,S_2(:,:,1),S_0(1))-gmm.Guass.Mean(:,1)*gmm.Guass.Mean(:,1)';
    gmm.Guass.Sigma(:,:,2) = bsxfun(@rdivide,S_2(:,:,2),S_0(2))-gmm.Guass.Mean(:,2)*gmm.Guass.Mean(:,2)';
    
end

function [gmm,Mu1,Mu2] = cgmmProcessOnline(gmm)
    [len,siz2] = size(gmm.dataSet);
    label = ceil(gmm.NB_Guass*rand(1,len));

    logP = [0,0,1];
    % Main loop of the online EM algorithm
    Mu1 = zeros(len,siz2);% plot the update
    Mu2 = zeros(len,siz2);% plot the update
    for Index2  = 1:gmm.onlineMaxIter
        for Index = 0:len
            %% E step
            if Index == 0
                [S_0,S_1,S_2] = EstepInit(gmm.NB_Guass,siz2);
            else                
                [S_0,S_1,S_2] = Estep(gmm.dataSet(Index,:)',gmm.Guass,S_0,S_1,S_2,(Index)^(-0.6));
            end
            %% M step
            if Index > 80
                [gmm] = Mstep(gmm,S_0,S_1,S_2);
            end
            %% for Plot the update 
                Mu1(Index+1,:) = gmm.Guass.Mean(:,1)';
                Mu2(Index+1,:) = gmm.Guass.Mean(:,2)';
            
        end
    end
end

function r = GaussMixRnd(MU1,SIGMA1,MU2,SIGMA2,ALPHA,N)
    % MU1 = [1 2];
    % SIGMA1 = [0.5 0; 0 .5];
    % MU2 = [-3 -5];
    % SIGMA2 = [1 0; 0 1];
    % ALPHA = 0.1;
    r1=mvnrnd(MU1,SIGMA1,N);
    r2=mvnrnd(MU2,SIGMA2,N);
    r = zeros(N,2);
    for i =1:N
        if rand(1) < ALPHA
            r(i,1)=r1(i,1);
            r(i,2)=r1(i,2);
        else
            r(i,1)=r2(i,1);
            r(i,2)=r2(i,2);
        end
    end
    figure;
    plot(r(:,1),r(:,2),'.');
end
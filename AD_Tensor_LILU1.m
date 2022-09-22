function [X,S,area0] = AD_Tensor_LILU1(Y,alphia,beta,tau,nu,mu,gamma,alpha,truncate_rank,maxIter,tol1,tol2,normal_map,anomaly_map)
%% argmin f(S,X)=tau/2*(||DhX1||_F^2+||DwX2||_F^2)+alphia*||X3||_tnnr+beta*||S3||_2,1   s.t. Y=X+S;
%% argmin f(V1,V2,V3,V4,S,X,A,B,D1,D2,D3,D4,D5,mu)=tau/2*(||DhX1||_F^2+||DwX2||_F^2)+alphia*(||X3||_*-A*X3*B'+beta*||S3||_2,1+0.5*mu(||Y-X-S+D1||_F^2+||V1-X+D2||_F^2+||V2-X+D3||_F^2+||V3-X+D4||_F^2++||V4-X+D5||_F^2)
%%
[H,W,Dim] = size(Y);
maxIter = 50;
mu1 = 0.0000001;
mu_bar = 1e10;
rho = 1.5;

%%
%---------------------------------------------
%  Initializations
%---------------------------------------------


X = zeros(H,W,Dim);
S=X;
%%%%aux variables
V1=X;
V2=X;
V3=X;
V4=V3;
% Lagrange Multipliers
D4=X;
D5=X;
D6=X;
A=zeros(Dim,max(truncate_rank,1));
B=zeros(H*W,max(truncate_rank,1));
sigma2=0;
area0=0;
HW=H*W;
y=reshape(Y,HW*Dim,1); 
s=reshape(S,HW*Dim,1);
b1=zeros(HW*Dim,1); 
x=s;b2=b1; p=b1;q=b1; 
% s=b1; 
j=b1; b3=b1;
% Make total variation matrix
Dh=TVmatrix(H,W,'H');
Dv=TVmatrix(H,W,'V');
Dd=opTV1(Dim);
Dd=Dd';
D1=kron(Dd',Dh); 
D2=kron(Dd',Dv);


% difference operators (periodic boundary)
D = @(z) cat(4, z([2:end, 1],:,:) - z, z(:,[2:end, 1],:)-z);
Dt = @(z) [-z(1,:,:,1)+z(end,:,:,1); - z(2:end,:,:,1) + z(1:end-1,:,:,1)] ...
    +[-z(:,1,:,2)+z(:,end,:,2), - z(:,2:end,:,2) + z(:,1:end-1,:,2)];
% for fftbased diagonilization
Lap = zeros(H,W);
Lap(1,1) = 4; Lap(1,2) = -1; Lap(2,1) = -1; Lap(end,1) = -1; Lap(1,end) = -1;
Lap = fft2(repmat(Lap, [1,1,Dim]));
B3 = reshape(b3,H,W,Dim);
v=D(B3);
w = v;
%%
%---------------------------------------------
%  AL iterations - main body
%---------------------------------------------
iter = 1;
res = inf*ones(5,1);

while (iter <= maxIter) && (sum(abs(res)) > tol2)
 %% update x
%      X = (Y-S+D4+D5+D6+V1+V2+V3)/4; 
%      x=reshape(X,HW*Dim,1);
    lastx=x;
    V3=reshape(V3,HW*Dim,1);
    D4=reshape(D4,HW*Dim,1);
    D6=reshape(D6,HW*Dim,1);
    bigY=tau*(y-s+D6)+nu*D1'*(p-b1)+nu*D2'*(q-b2)+nu*(j-b3)+mu1*(V3-D4);        
    [x,~]=lsqr(@afun,bigY,1e-15,10,[],[],x);  
    V3=reshape(V3,H,W,Dim);
    D4=reshape(D4,H,W,Dim); 
    D6=reshape(D6,H,W,Dim);
   %% update p,V1
    p=SoftTh(D1*x+b1,mu/nu);
    V1=fold_k(p,1,[H,W,Dim]);
    %V1=real(ifft2_slice(fft2_slice(mu*(X-D2))./denh));
%% update  q,V2   
     q=SoftTh(D2*x+b2,mu/nu);
     V2=fold_k(q,1,[H,W,Dim]);
     %V2=real(ifft2_slice(fft2_slice(mu*(X-D3))./denw));
%% update j
    B3=reshape(b3,H,W,Dim);
    X=reshape(x,H,W,Dim);
    rhs = nu*(X+B3) + Dt(v-w)/gamma;
    J = real((ifftn(fftn(rhs))./(Lap/gamma+nu))); 
    j=reshape(J,HW*Dim,1);
%% update v
    v = ProjL10ball(D(J)+w,alpha);

%% update V3
    temp = 0.5*(X+V4-D4+D5);
    temp3=Tensor_unfold(temp,3)';
    [Us,sigma,Vs] = svd(temp3,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>(alphia/(2*mu1))));
    if svp >= 1
        sigma = sigma(1:svp)-(alphia/(2*mu1));
    else
        svp = 1;
        sigma = 0;
    end
    V3_3 = Us(:,1:svp)*diag(sigma)*Vs(:,1:svp)';  
    V3=fold_k(V3_3',3,[H,W,Dim]);
   %% update V4
   temp3=fold_k(alphia*B*A'/mu1,3,[H,W,Dim]);
   V4=V3-D5+temp3;
   %% update A ,B
   temp3=Tensor_unfold(V4,3)';
 
   if truncate_rank>=1
%        [A,sigma2,B]=lansvd(temp3,truncate_rank,'L');
%    else
       [A,sigma2,B]=svds(temp3,truncate_rank);
   end
   if sigma2(1,1)==0 || truncate_rank==0
       A=zeros(Dim,max(truncate_rank,1));
       B=zeros(H*W,max(truncate_rank,1));
   end

    %% update S
    S = solve_l1l1l2(Y-X+D6,beta/mu1);
   % figure(5),imshow(sum(abs(X),3),[]);
    %% update D1,D2,D3,D4,D5
     D6=D6+(Y-X-S);
     D4=D4-X+V3;
     D5=D5-V3+V4;
     b1=b1+D1*x-p;
     b2=b2+D2*x-q;
     b3=b3+x-j;
     w = w + D(J) - v;
     %%
     if mod(iter,10) == 1      
        t0=Y-X-S;
        t0=t0(:);
        res(1)=sqrt(t0'*t0);
%         t1=p-D1*x;
        t1=X-V1;
        t1=t1(:);
        res(2)=sqrt(t1'*t1);
%         t2=q-D2*x;
        t2=X-V2;
        t2=t2(:);
        res(3)=sqrt(t2'*t2);
        t3=X-V3;
        t3=t3(:);
        res(4)=sqrt(t3'*t3);
        t4=V3-V4;
        t4=t4(:);
        res(5)=sqrt(t4'*t4);

        f_show=sqrt(sum(S.^2,3));
        r_max = max(f_show(:));
        taus = linspace(0, r_max, 5000);
        for index2 = 1:length(taus)
          tau1 = taus(index2);
          anomaly_map_rx = (f_show(:)> tau1)';
          PF0(index2) =sum(anomaly_map_rx & normal_map)/sum(normal_map);
          PD0(index2) =sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
        end
        id=(iter-1)/10+1;
         area0(id)=sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
         RES(id)=res(1);
        disp(['iter =',num2str(iter),'- res(1) =',num2str(res(1)), ',res(2) =',num2str(res(2)),',res(3) =', num2str(res(3)),',res(4) =', num2str(res(4)),',res(5) =', num2str(res(5)),',AUC=',num2str(area0(id))]);
%   disp(['iter =',num2str(iter),'- res(1) =',num2str(res(1)), ',res(2) =',num2str(res(2)),',res(3) =', num2str(res(3)),',res(4) =', num2str(res(4)),',res(5) =', num2str(res(5))]);
    end  
    iter = iter + 1;    
    mu1 = min(mu1*rho, mu_bar); 
end
% f_show=sqrt(sum(S.^2,3));
% r_max = max(f_show(:));
% taus = linspace(0, r_max, 5000);
% for index2 = 1:length(taus)
%   tau1 = taus(index2);
%   anomaly_map_rx = (f_show(:)> tau1)';
%   PF0(index2) =sum(anomaly_map_rx & normal_map)/sum(normal_map);
%   PD0(index2) =sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% end
% area0=sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
 area0=area0(end);
%% This is a function handle used in LSQR
 function y = afun(x,str)
       tempval= nu*((D1'*(D1*x))+(D2'*(D2*x)))+ tau*x +nu*x;
            switch str
                case 'transp'
                    y = tempval;
                case 'notransp'
                    y = tempval;
            end
 end

end

function [E] = solve_l1l1l2(X,lambda)
[H,W,D] = size(X);
nm=sqrt(sum(X.^2,3));
nms=max(nm-ones(H,W)*lambda,0);
sw=repmat(nms./nm,[1,1,D]);
E=sw.*X;
end


%% Soft Thresholding
function X=SoftTh(B,lambda)
      
       X=sign(B).*max(0,abs(B)-(lambda/2));
       
end

%% Total Variation
function opD=TVmatrix(m,n,str)

if str=='H' % This will give matrix for Horizontal Gradient
    D = spdiags([-ones(n,1) ones(n,1)],[0 1],n,n);
    D(n,:) = 0;
    D = kron(D,speye(m));
   
elseif str=='V' %This will give matrix for Verticle Gradient
   D = spdiags([-ones(m,1) ones(m,1)],[0 1],m,m);
   D(m,:) = 0;
   D = kron(speye(n),D);
end
opD=D;

end
%% 
function opD=opTV1(m)

%Make two vectors of elements -1 and 1 of lengths (m-1)
B=[ -1*ones(m-1,1),ones(m-1,1)]; 
%Make sparse diagonal matrix D of size m-1 by m
%with -1's on zeroth diagonal and 1's on 1st diagonal;
D=spdiags(B,[0,1],m-1,m);
%add a last row of all zeros in D
D(m,:)=zeros(1,m);
%Make it as operator
opD=D; %It will convert to operator.
end

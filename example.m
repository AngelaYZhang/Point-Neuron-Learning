%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%    Point Neuron Example Manuscript
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creation   : 15-09-24
% Author     : Hanwen Bi (hanwen.bi@anu.edu.au)
% Version    : 21-09-24
% Descirption:
%       Using Point Neuron to estimate the reverberant sound field over a  
%       2D circular region from the microphone measurement randomly 
%       displaced over that region.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%    Basic Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clear 
rng('default');
rng(1);


Frequency   = 900;
c           = 343;
k           = 2*pi*Frequency/c;
% Radius of the target region
TargetR     = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%    Set Microphone 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% Load Microphone coordinate and microphone measured sound field
load('Mic75_random_Freq900.mat')
% Number of microphones
MicN = length(MICCAR(:,1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%    Mic Soundfield
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% Add WGN to measurement 
MicPrimaryField  = awgn((MicPrimaryField),30,'measured');
% Normalization factor of the microphone measurement
NormalFactor     = norm(MicPrimaryField);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%    Point Neuron Processing 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

%%%%%%%% Setting 
%%%% Load inital Point Neuron coordinate
load('PNinital_Freq900.mat')
%%%% Point Neuron number
PnN     = length(InCoord);
%%%% Iteration number
IterN   = 20000;
%%%% Setp size for point neruon weight
StepW   = 0.03;
%%%% Setp size for point neruon coordinate
StepC   = 0.005;
%%%% Penalty for Sparsity
Lambda  = 0.0005;
%%%%%% Initialization 
% Randomly initialize the point neuron weight. Note: Other method can be
% applied as well, we take random initialization as an example
InWeight = rand(PnN,1);
%%%%%% Run Point Neuron Network
% Note the parameters are not optimal, just take a an example
% Please feel free to try with different initializations and parameters
[PnCoord, PnWeight,PnWscale, Loss] = PointNeuron(k, MicPrimaryField/ ...
    NormalFactor,[MICCAR,zeros(MicN,1)],[InCoord,zeros(PnN,1)], ...
    InWeight,StepW,[StepC,StepC,0],Lambda,IterN);

%%%% De-normalization 
PnWeight = PnWeight*NormalFactor;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%    Loss Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
figure()
plot(Loss)
title('Loss')
xlabel('Iteration number')
ylabel('Loss')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%    Sound Field Estimation Analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% uniformly sample the plane
PxN           = 50;
PyN           = 50;
SizeX         = 1.3;
SizeY         = 1.3;

PxSamping     = linspace(-SizeX,SizeX,PxN);
PySamping     = linspace(-SizeY,SizeY,PyN);

PlanePn       = zeros(PxN*PyN,1);
TargetPn      = [];

Index         = 1;
IndexT        = 1;
for i=1:PxN
    for j=1:PyN                
        SampleV          = [PxSamping(i);PySamping(j)];                                                                     
        WavePn           = [PnCoord(:,1).';PnCoord(:,2).'];    
        DisPn            = sum((WavePn-SampleV).^2,1);
        % Point neuron estimation           
        PlanePn(Index,1) = ((PnWeight.').*(PnWscale.'))* ...
                           (exp(1i*k*sqrt(DisPn))./(4*pi*sqrt(DisPn))).';
                            
        % The sample is in the target region or not?                                  
        if PxSamping(i)^2+PxSamping(j)^2<TargetR^2
            TargetPn(IndexT,1) = PlanePn(Index,1);
            IndexT             = IndexT+1;
        end        
        Index            = Index+1;       
    end
end

%%%%%%%%% Load ground truth
load(['Cir1124_Freq',num2str(Frequency),'.mat'])
load(['Pla2500_Freq',num2str(Frequency),'.mat'])

% Error per sample point
EME          = 10*log10((abs(PlaneSure-PlanePn).^2)./(abs(PlaneSure).^2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%   Sound Field Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% Plot settings
Resolution  = 0.02;
[Px,Py]     = meshgrid(-SizeX:Resolution:SizeX,-SizeY:Resolution:SizeY);

% Original sound field for plot
SpS         = griddata(PLACAR(:,1),PLACAR(:,2),PlaneSure,Px,Py);
% Sound field by Point neuron for plot                              
SpPn        = griddata(PLACAR(:,1),PLACAR(:,2),PlanePn,Px,Py);
% Error data per sample point
ErrorMapE   = griddata(PLACAR(:,1),PLACAR(:,2),EME,Px,Py);

% plot the target region boundary
targetX     = cos(0:0.005:2*pi);
targetY     = sin(0:0.005:2*pi);

%%%%%%%%%% Plot original sound field                             
figure()

pcolor(Px,Py,real(SpS));
shading interp;
colorbar;
xlim([-1.2,1.2]);
ylim([-1.2,1.2]);
caxis([-0.25,0.25]);
xlabel('x (m)')
ylabel('y (m)')
tickvals = [-1.2, -0.8, -0.4,0, 0.4,0.8,1.2];
set(gca,'YTick',tickvals);
set(gca,'XTick',tickvals);
hold on
plot(targetX,targetY,'-','Color',[0,0,0],'LineWidth',1)
title('source sound field')
ax = gca;

[cmin, cmax] = deal(ax.CLim);

%%%%%%%%%%%% Plot point neuron estimated sound field
figure()

pcolor(Px,Py,real(SpPn));
shading interp;
colorbar;
xlim([-1.2,1.2]);
ylim([-1.2,1.2]);
caxis([-0.25,0.25]);
xlabel('x (m)')
ylabel('y (m)')
tickvals = [-1.2, -0.8, -0.4,0, 0.4,0.8,1.2];
set(gca,'YTick',tickvals);
set(gca,'XTick',tickvals);
xlabel('x (m)')
ylabel('y (m)')
hold on
plot(targetX,targetY,'-','Color',[0,0,0],'LineWidth',1)
title('Point neuron estimated sound field')

%%%%%%%%%%%% Plot error map


figure()
clim=[-15,5];

imagesc(Py(:,1),Py(:,1),ErrorMapE,clim)
colormap hot
xlabel('x (m)')
ylabel('y (m)')
tickvals = [-1.2, -0.8, -0.4,0, 0.4,0.8,1.2];
set(gca,'YTick',tickvals);
set(gca,'XTick',tickvals);
colorbar;
c            =colorbar;
c.Label.String='dB';
hold on
plot(targetX,targetY,'-','Color',[0,0,0],'LineWidth',1)
title('Error Map') 

%%%%%%%%%%% Calculate NMSE
% Error over the plane                      
ErrorPlane   = 20*log10(norm(PlaneSure-PlanePn)/norm(PlaneSure));
% Error over the target region
ErrorTarget  = 20*log10(norm(ValueSure-TargetPn)/norm(ValueSure));



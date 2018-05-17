clear all
% Particular configuration
mul=[0.4,0.6,5,1.2,1,0.1];
% load SN.DAT
% nb = 4;
% mul = [SN(nb,1:end-1) 1 SN(nb,end)];
% mul=[1.8 4.2 5.7 1.9 1 0.3];
load Grids
Ah=sparse(coarse.nodes,coarse.nodes);
% Call FEM for calculating Aq and Fh
[Aq, Fh] = FEM(coarse);
% save FEMMatrices Aq Fh
for i=1:6
    Ah=Ah+mul(i)*Aq{i};
end
clear Aq
% Solve for uh and Troot
uh=Ah\Fh;
Troot= Fh'*uh;
% Plot the solution
%figure
%plotsolution(coarse,uh);

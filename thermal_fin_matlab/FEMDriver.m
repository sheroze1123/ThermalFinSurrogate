clear 
% Particular configuration
%mul=[0.4,0.6,5,1.2,1,0.1];
% load SN.DAT
% nb = 4;
% mul = [SN(nb,1:end-1) 1 SN(nb,end)];

% mul=[1.8 4.2 5.7 1.9 1 0.3];

load grids
% Call FEM for calculating Aq and Fh
[Aq, Fh] = FEM(coarse);

training_data_pts = 200;
muls = zeros(training_data_pts, 6);
uh_len = length(Fh);
uhs = zeros(training_data_pts, uh_len);

for j = 1:training_data_pts

    mul=[rand*8 rand*8 rand*8 rand*8 1 rand*2];
    muls(j,:) = mul;

    Ah=sparse(coarse.nodes,coarse.nodes);

    % save FEMMatrices Aq Fh
    for i=1:6
        Ah=Ah+mul(i)*Aq{i};
    end

    % Solve for uh and Troot
    uh=Ah\Fh;
    Troot= Fh'*uh;

    % Plot the solution
    %plotsolution(coarse,uh,1);
    
    uhs(j,:) = uh;
end

dlmwrite('training_data_mul.csv', muls);
dlmwrite('training_data_uh.csv', uhs);



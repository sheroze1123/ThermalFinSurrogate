%FEM.m
%Author: Unknown

function [Aq,Fh,M] = FEM(grid)
% This function will take the configuration and a triangulation and return Aq and Fh
% by using finite element method

% Global node
Node = zeros(3,1);
% Global coordinates
x = zeros(3,1);
y = zeros(3,1);
% Elemental matrices I
AI = zeros(3, 3);
% Right hand side vector F
Fh = zeros(grid.nodes,1);
% Assembly Matrix Aq from AI
M = sparse(grid.nodes,grid.nodes);
Am = [2 1 1;1 2 1; 1 1 2];
nd = length(grid.theta);
for n = 1:(nd-2)
    Aq{n} = sparse(grid.nodes,grid.nodes);
    % Number of elements on each region
    m = size(grid.theta{n},1);
    for k = 1:m
        % Take global node for current element
        Node(1) = grid.theta{n}(k, 1);
        Node(2) = grid.theta{n}(k, 2);
        Node(3) = grid.theta{n}(k, 3);
        % x-y coordinates for each global node
        for i = 1:3
            x(i) = grid.coor(Node(i),1);
            y(i) = grid.coor(Node(i),2);
        end
        % The area of current element
        Area = 0.5*abs(x(2)*y(3)-y(2)*x(3)-x(1)*y(3)+y(1)*x(3)+x(1)*y(2)-y(1)*x(2));
        % Calculate cx and cy
        cx(1) = y(2) - y(3);
        cx(2) = y(3) - y(1);
        cx(3) = y(1) - y(2);
        cy(1) = x(3) - x(2);
        cy(2) = x(1) - x(3);
        cy(3) = x(2) - x(1);
        % establish the elemental matrices
        for i = 1:3
            for j = 1:3
                AI(i,j) = cx(i)*cx(j) + cy(i)*cy(j);
            end
        end
        AI = AI/(4*Area);
        
        % Element Mass matrix
        AM = (Area/12)*Am;
        
        % Assembly matrix A
        for alpha = 1:3
            i = Node(alpha);
            for beta = 1:3
                j = Node(beta);
                 Aq{n}(i,j) = Aq{n}(i,j) + AI(alpha, beta);
                 M(i,j) = M(i,j) + AM(alpha,beta);
            end
        end
    end
end

% Elemental matrices II
AII = zeros(2, 2);
Node = zeros(2,1);
x = zeros(2,1);
y = zeros(2,1);
% Assembly Matrix A from elemental matrices AII and RHS
Aq{nd-1} = sparse(grid.nodes,grid.nodes);
for m=(nd-1):nd
    % The number of segments on the boundary excluding the Root
    n = size(grid.theta{m},1);
    for k = 1:n
        % Global node of current segment
        Node(1) = grid.theta{m}(k, 1);
        Node(2) = grid.theta{m}(k, 2);
        % x-y coordinates of global nodes
        for i = 1:2
            x(i) = grid.coor(Node(i),1);
            y(i) = grid.coor(Node(i),2);
        end
        % Length of current segment
        h = sqrt((x(1) - x(2))^2 + (y(1) - y(2))^2);  
        if m==(nd-1)
            % Establish the elemental matrices II
            AII =(h/3)*[1 1/2; 1/2 1];
            % Assembly matrix A
            for alpha = 1:2
                i = Node(alpha);
                for beta = 1:2
                    j = Node(beta);
                    Aq{m} (i,j) = Aq{m}(i,j) + AII(alpha, beta);
                end
            end
        else
            % Establish the elemental matrices for RHS
            Fe = h/2*[1 1];
            % Assembly the RHS
            for alpha = 1:2
                i = Node(alpha);
                Fh(i) = Fh(i) + Fe(alpha);
            end
        end       
    end
end

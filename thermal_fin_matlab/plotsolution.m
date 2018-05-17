function plotsolution( grid, u ,flag)
% This function plots the temperature distribution in the grid...
% Grid: Is one of the grids
% u: Is the solution vector...


% axis off;
figure; hold on
if flag == 0
    caxis([min(u) max(u)]);
else
    caxis([-0.14 0.03]);
end

for i=1:7
  for j=grid.theta{i}'
    fill(grid.coor(j,1),grid.coor(j,2),u(j));
  end
end

colormap(jet)
colorbar;

shading interp;

hold off;


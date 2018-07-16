function plotsolution( grid, u ,flag, id)
% This function plots the temperature distribution in the grid...
% Grid: Is one of the grids
% u: Is the solution vector...


% axis off;
%gcf = figure('visible','off'); hold on
figure; hold on
if flag == 0
    caxis([min(u) max(u)]);
else
    caxis([-0.14 0.03]);
end

for i=1:7
  for j=grid.theta{i}'
    %grid.theta are the triangle indices
    fill(grid.coor(j,1),grid.coor(j,2),u(j));
    
  end
end
fill(-2.7,2.4,3);
fill(-2.7,2.3,1);
fill(-2.6,2.4,3);
%colormap('gray')
%ax = gca;
%ax.Visible = 'off';
colorbar;
%set(gcf,'PaperUnits','inches','PaperPosition',[0 0 1 1]);
%shading interp;
%shading interp
%saveas(gcf,"fin_" + id + ".png")
%disp("Saved image " + id);
hold off;


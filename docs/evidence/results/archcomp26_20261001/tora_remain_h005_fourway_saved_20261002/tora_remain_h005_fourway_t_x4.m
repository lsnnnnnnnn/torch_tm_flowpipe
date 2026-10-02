% 2026 TORA remain h=0.05 supplemental saved-box visualization; no solver run.
% Reads adjacent geometry JSON. MATLAB execution was not part of generation.
base = fileparts(mfilename('fullpath'));
g = jsondecode(fileread(fullfile(base,'tora_remain_h005_fourway_t_x4.geometry.json')));
t = 0:g.step_s:g.steps*g.step_s; figure('Color','w'); hold on; grid on;
patch([0 20 20 0],[-2 -2 2 2],[0.83 0.92 0.84],'FaceAlpha',0.25,'EdgeColor','none','DisplayName','Safe x4 band');
for j = 1:numel(g.series)
  s = g.series(j); b = s.tube_x4_union_per_step; c = sscanf(s.color(2:end),'%2x%2x%2x')'/255;
  stairs(t,[b(:,1);b(end,1)],'Color',c,'LineStyle',s.line_style,'LineWidth',1.2,'DisplayName',s.label);
  stairs(t,[b(:,2);b(end,2)],'Color',c,'LineStyle',s.line_style,'LineWidth',1.2,'HandleVisibility','off');
end
plot([0 0],[0.5 0.6],'ko-','DisplayName','Initial x4');
xlabel('t (s)'); ylabel('x4'); title('TORA remain h=0.05: four saved whole-step x4 tubes');
xlim([0 20]); ylim([-2.1 2.1]); legend('Location','best'); hold off;

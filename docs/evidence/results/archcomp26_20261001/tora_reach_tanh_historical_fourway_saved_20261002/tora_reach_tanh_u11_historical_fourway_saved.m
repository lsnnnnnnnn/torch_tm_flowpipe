% Historical TORA reach-tanh saved ranges; drawing only, no solver invocation.
% MATLAB script reads the adjacent geometry JSON. It has not been executed here.
base = fileparts(mfilename('fullpath'));
g = jsondecode(fileread(fullfile(base, 'tora_reach_tanh_u11_historical_fourway_saved.geometry.json')));
t = 0:g.step_s:g.steps*g.step_s; figure('Color','w','Position',[100 100 1300 760]);
for dim = 1:2
  subplot(2,2,2*dim-1); hold on; grid on;
  if dim == 1, key = 'tube_x1'; target = g.target_at_T5_x1_x2.x1; else, key = 'tube_x2'; target = g.target_at_T5_x1_x2.x2; end
  plot([5 5],target,'m-','LineWidth',5,'DisplayName','T=5 target');
  for j = 1:numel(g.series)
    s = g.series(j); b = s.(key); c = sscanf(s.color(2:end),'%2x%2x%2x')'/255;
    stairs(t,[b(:,1);b(end,1)],'Color',c,'LineStyle',s.line_style,'LineWidth',1.6,'DisplayName',s.label);
    stairs(t,[b(:,2);b(end,2)],'Color',c,'LineStyle',s.line_style,'LineWidth',1.6,'HandleVisibility','off');
  end
  xlabel('t (s)'); ylabel(sprintf('x%d',dim)); xlim([0 5]); title(sprintf('Saved whole-step tube: x%d',dim));
  if dim == 1, legend('Location','best'); end
  subplot(2,2,2*dim); hold on; grid on;
  if dim == 1, endpoint_key = 'endpoint_T5_x1'; else, endpoint_key = 'endpoint_T5_x2'; end
  ref = g.series(1).(endpoint_key);
  for j = 1:numel(g.series)
    s = g.series(j); e = s.(endpoint_key); c = sscanf(s.color(2:end),'%2x%2x%2x')'/255;
    plot((e(1)-ref(1))*1e5,j,'o','Color',c,'MarkerFaceColor',c,'MarkerSize',7);
    plot((e(2)-ref(2))*1e5,j,'^','Color',c,'MarkerFaceColor',c,'MarkerSize',7);
  end
  xline(0,'k:'); yticks(1:numel(g.series)); yticklabels({g.series.label}); ylim([0.5 numel(g.series)+0.5]);
  xlabel('Endpoint bound offset vs Huan (10^{-5})'); title(sprintf('T=5 x%d bounds: circle lower, triangle upper',dim));
end
sgtitle('Historical TORA reach-tanh, official u=11f: saved numeric tubes');

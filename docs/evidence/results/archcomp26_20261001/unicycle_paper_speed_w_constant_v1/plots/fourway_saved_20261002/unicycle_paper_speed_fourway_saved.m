% Saved 2026 Unicycle paper-speed-constant-w four-way figure. No solver is run.
% Reads adjacent geometry JSON. MATLAB execution was not part of generation.
base = fileparts(mfilename('fullpath'));
g = jsondecode(fileread(fullfile(base,'unicycle_paper_speed_fourway_saved.geometry.json')));
t = 0:g.step_s:g.full_horizon_s; figure('Color','w','Position',[100 100 1350 800]);
for j = 1:2
  if j == 1, key = 'tube_x1'; state = 1; else, key = 'tube_x3'; state = 3; end
  subplot(2,2,2*j-1); hold on; grid on;
  target = g.target_at_T10_by_physical_state.(sprintf('x%d',state));
  plot([10 10],target,'m-','LineWidth',4,'DisplayName','T=10 target');
  for k = 1:numel(g.series)
    s = g.series(k); b = s.(key); c = sscanf(s.color(2:end),'%2x%2x%2x')'/255;
    for q = 1:g.steps, patch([t(q) t(q+1) t(q+1) t(q)], [b(q,1) b(q,1) b(q,2) b(q,2)], c, 'FaceAlpha',0.035,'EdgeColor','none'); end
    stairs(t,[b(:,1);b(end,1)],'Color',c,'LineStyle',s.line_style,'LineWidth',1.2,'DisplayName',s.label);
    stairs(t,[b(:,2);b(end,2)],'Color',c,'LineStyle',s.line_style,'LineWidth',1.2,'HandleVisibility','off');
  end
  xlabel('t (s)'); ylabel(sprintf('x%d',state)); xlim([0 10]);
  title(sprintf('Saved whole-step x%d tube',state)); if j == 1, legend('Location','best'); end
end
for j = 1:2
  if j == 1, state = 3; else, state = 4; end
  subplot(2,2,2*j); hold on; grid on;
  target = g.target_at_T10_by_physical_state.(sprintf('x%d',state));
  patch([target(1) target(2) target(2) target(1)],[0.5 0.5 4.5 4.5],[0.83 0.92 0.84],'FaceAlpha',0.35,'EdgeColor','none');
  for k = 1:numel(g.series)
    s = g.series(k); e = s.terminal_physical_endpoint(state,:); c = sscanf(s.color(2:end),'%2x%2x%2x')'/255;
    plot(e,[k k],'-o','Color',c,'LineWidth',3,'MarkerFaceColor',c);
  end
  plot([target(1) target(1)],[0.5 4.5],'k--');
  yticks(1:numel(g.series)); yticklabels({g.series.label}); ylim([0.5 4.5]);
  if state == 4, xlim([-0.35 -0.18]); xticks([-0.35 -0.30 -0.25 -0.20]); end
  xlabel(sprintf('T=10 endpoint x%d',state)); title(sprintf('Target x%d: [%g,%g]',state,target));
end
sgtitle('Unicycle paper-speed constant-w: saved four-method numeric comparison');

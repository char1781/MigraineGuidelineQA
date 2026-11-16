/* migraine-rag-eval-report · assets/charts.js
 * 所有图表逻辑放这里，HTML 里只留 <div> 容器。
 */
(function() {
  var style = getComputedStyle(document.documentElement);
  var accent  = style.getPropertyValue('--accent').trim();   // #2563eb
  var accent2 = style.getPropertyValue('--accent2').trim();  // #059669
  var warn    = style.getPropertyValue('--warn').trim();
  var danger  = style.getPropertyValue('--danger').trim();
  var ink     = style.getPropertyValue('--ink').trim();
  var muted   = style.getPropertyValue('--muted').trim();
  var rule    = style.getPropertyValue('--rule').trim();
  var bg2     = style.getPropertyValue('--bg2').trim();

  var palette = [
    '#2563eb',   // ES
    '#059669',   // Hybrid
    '#d97706',   // Graph
    '#7c3aed',   // Web
  ];

  // ─── 图1: 四通道指标对比（分组柱状图） ───
  var chart1 = echarts.init(document.getElementById('chart-channel-compare'), null, { renderer: 'svg' });
  chart1.setOption({
    tooltip: {
      trigger: 'axis',
      appendToBody: true,
      valueFormatter: function(v) { return Number(v).toFixed(3); }
    },
    legend: {
      data: ['ES 40', 'Hybrid 40', 'Graph 10', 'Web 10'],
      bottom: 0,
      textStyle: { color: muted, fontSize: 12 }
    },
    grid: { top: 20, left: 50, right: 20, bottom: 60 },
    xAxis: {
      type: 'category',
      data: ['Faithfulness', 'Answer\nRelevancy', 'Context\nPrecision', 'Context\nRecall'],
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11, lineHeight: 14 }
    },
    yAxis: {
      type: 'value',
      min: 0.2,
      max: 1.0,
      interval: 0.2,
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } },
      axisLabel: { color: muted, fontSize: 11 }
    },
    series: [
      {
        name: 'ES 40', type: 'bar', barWidth: 16, barGap: '-100%',
        itemStyle: { borderRadius: [4,4,0,0], color: palette[0] },
        data: [0.866, 0.758, 0.766, 0.800]
      },
      {
        name: 'Hybrid 40', type: 'bar', barWidth: 16,
        itemStyle: { borderRadius: [4,4,0,0], color: palette[1] },
        data: [0.939, 0.928, 0.907, 0.975]
      },
      {
        name: 'Graph 10', type: 'bar', barWidth: 16,
        itemStyle: { borderRadius: [4,4,0,0], color: palette[2] },
        data: [0.992, 0.855, 0.900, 0.317]
      },
      {
        name: 'Web 10', type: 'bar', barWidth: 16,
        itemStyle: { borderRadius: [4,4,0,0], color: palette[3] },
        data: [0.878, 0.783, 0.717, 0.700]
      }
    ],
    animation: false
  });
  window.addEventListener('resize', function() { chart1.resize(); });

  // ─── 图2: 各通道 Overall 排名（水平条形图） ───
  var chart2 = echarts.init(document.getElementById('chart-overall-rank'), null, { renderer: 'svg' });
  // 先按分数降序排
  var channels = [
    { name: 'Hybrid (40条)', value: 0.937, color: palette[1] },
    { name: 'ES (40条)',     value: 0.798, color: palette[0] },
    { name: 'Web (10条)',   value: 0.770, color: palette[3] },
    { name: 'Graph (10条)', value: 0.766, color: palette[2] }
  ].sort(function(a,b) { return a.value - b.value; });

  chart2.setOption({
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'shadow' }, appendToBody: true,
      valueFormatter: function(v) { return 'Overall: ' + Number(v).toFixed(3); }
    },
    grid: { top: 20, left: 120, right: 60, bottom: 20 },
    xAxis: {
      type: 'value', min: 0.6, max: 1.0, interval: 0.1,
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } },
      axisLabel: { color: muted, fontSize: 11 }
    },
    yAxis: {
      type: 'category',
      data: channels.map(function(c) { return c.name; }),
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false },
      axisLabel: { color: ink, fontSize: 12, fontWeight: 600 }
    },
    series: [{
      type: 'bar',
      barWidth: 24,
      itemStyle: {
        borderRadius: [0, 5, 5, 0],
        color: function(params) { return channels[params.dataIndex].color; }
      },
      label: {
        show: true, position: 'right',
        formatter: function(p) { return p.value.toFixed(3); },
        color: ink, fontSize: 12, fontWeight: 600
      },
      data: channels.map(function(c) { return c.value; })
    }],
    animation: false
  });
  window.addEventListener('resize', function() { chart2.resize(); });

  // ─── 图3: 修复前后关键指标对比 ───
  var chart3 = echarts.init(document.getElementById('chart-before-after'), null, { renderer: 'svg' });
  chart3.setOption({
    tooltip: {
      trigger: 'axis', appendToBody: true,
      valueFormatter: function(v) { return Number(v).toFixed(3); }
    },
    legend: {
      data: ['修复前 (11:42)', '修复后 (现在)', '变化量'],
      bottom: 0,
      textStyle: { color: muted, fontSize: 12 }
    },
    grid: { top: 20, left: 60, right: 40, bottom: 70 },
    xAxis: {
      type: 'category',
      data: ['Faithfulness', 'Answer\nRelevancy', 'Context\nPrecision', 'Context\nRecall', '**OVERALL**'],
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11, lineHeight: 14 }
    },
    yAxis: {
      type: 'value',
      min: 0.4,
      max: 1.0,
      interval: 0.1,
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } },
      axisLabel: { color: muted, fontSize: 11 }
    },
    series: [
      {
        name: '修复前 (11:42)', type: 'bar', barWidth: 18,
        itemStyle: { borderRadius: [4,4,0,0], color: '#94a3b8' },
        data: [0.815, 0.691, 0.680, 0.568, 0.689]
      },
      {
        name: '修复后 (现在)', type: 'bar', barWidth: 18,
        itemStyle: { borderRadius: [4,4,0,0], color: accent },
        data: [0.909, 0.838, 0.831, 0.812, 0.847]
      },
      {
        name: '变化量', type: 'line', smooth: false,
        itemStyle: { color: accent2 },
        symbol: 'circle', symbolSize: 8,
        label: {
          show: true,
          position: 'top',
          formatter: function(p) {
            var diffs = [0.094, 0.147, 0.151, 0.244, 0.158];
            return '+' + diffs[p.dataIndex].toFixed(3);
          },
          color: accent2, fontSize: 11, fontWeight: 700
        },
        data: [0.094, 0.147, 0.151, 0.244, 0.158]
      }
    ],
    animation: false
  });
  window.addEventListener('resize', function() { chart3.resize(); });

})();

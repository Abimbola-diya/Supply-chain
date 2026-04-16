'use client';

import React, { useEffect, useRef, useState, useMemo, useCallback, useLayoutEffect } from 'react';
import {
  geoMercator,
  geoPath,
  select,
  range,
  scaleLinear,
  extent,
  max,
  min,
} from 'd3';
import { useStore } from '@/stores';
import { Node, Flow } from '@/types';

interface MapViewProps {
  nodes: Node[];
  flows: Flow[];
}

// Professional color palette - matches reference designs
const COLORS = {
  upstream: '#ff6b35',
  refinery: '#c77dff',
  terminal: '#ec4899',
  importer: '#0ea5e9',
  depot: '#fbbf24',
  retail: '#10b981',
  fpso: '#f97316',
  background: '#0a0e17',
  surface1: '#0f1520',
  surface2: '#141b28',
  gridLight: 'rgba(107, 122, 150, 0.08)',
  textPrimary: '#f8fafc',
  textSecondary: '#b0c0dd',
  textMuted: '#6b7a96',
  danger: '#ef4444',
  warning: '#f97316',
  success: '#22c55e',
  accent: '#0ea5e9',
};

// Map type config - node type to color mapping
const NODE_TYPE_COLORS: Record<string, string> = {
  upstream_field: COLORS.upstream,
  upstream_ep: COLORS.upstream,
  nnpc_ep: COLORS.upstream,
  refinery: COLORS.refinery,
  terminal: COLORS.terminal,
  export_terminal: COLORS.terminal,
  jetty: COLORS.terminal,
  fpso: COLORS.fpso,
  importer: COLORS.importer,
  import_point: COLORS.importer,
  depot: COLORS.depot,
  distribution_center: COLORS.depot,
  retail_station: COLORS.retail,
  retail_network: COLORS.retail,
  storage: COLORS.depot,
  pipeline: COLORS.textMuted,
};

export default function MapViewD3({ nodes, flows }: MapViewProps) {
  const {
    camera,
    set_camera,
    enabled_node_types,
    enabled_statuses,
    selected_node,
    set_selected_node,
    set_hovered_node,
    highlighted_nodes,
    ai_focus_nodes,
    active_layers,
  } = useStore();

  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ node: Node; x: number; y: number } | null>(null);
  const [panning, setPanning] = useState(false);
  const [startPan, setStartPan] = useState({ x: 0, y: 0 });
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(2.5);

  // Filter visible nodes
  const visibleNodes = useMemo(
    () =>
      nodes.filter(
        (n) =>
          enabled_node_types.has(n.node_type.toLowerCase()) &&
          enabled_statuses.has(n.status.toLowerCase())
      ),
    [nodes, enabled_node_types, enabled_statuses]
  );

  // Build flow graph
  const flowData = useMemo(() => {
    if (!active_layers.has('flows')) return [];
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));
    return flows
      .filter((f) => nodeMap.has(f.source_node_id) && nodeMap.has(f.destination_node_id))
      .map((f) => {
        const src = nodeMap.get(f.source_node_id)!;
        const dst = nodeMap.get(f.destination_node_id)!;
        return {
          id: f.id,
          source: src,
          target: dst,
          volume: f.volume_bpd || 1000,
          mode: f.transport_mode || 'pipeline',
          product: f.product_type || 'unknown',
        };
      });
  }, [flows, nodes, active_layers]);

  // Get node color based on type and status
  const getNodeColor = useCallback(
    (node: Node) => {
      if (node.id === selected_node) return COLORS.accent;
      if (highlighted_nodes.has(node.id) || ai_focus_nodes.includes(node.id))
        return '#fbbf24';
      const baseColor = NODE_TYPE_COLORS[node.node_type.toLowerCase()] || COLORS.textSecondary;
      return baseColor;
    },
    [selected_node, highlighted_nodes, ai_focus_nodes]
  );

  // Get node radius based on capacity
  const getNodeRadius = useCallback(
    (node: Node) => {
      const isSelected = node.id === selected_node;
      const isHighlighted = highlighted_nodes.has(node.id) || ai_focus_nodes.includes(node.id);
      const base = Math.min(4 + Math.log(Math.max(node.capacity_bpd || 50000, 1)) / 2, 12);
      return base * (isSelected ? 1.8 : isHighlighted ? 1.4 : 1);
    },
    [selected_node, highlighted_nodes, ai_focus_nodes]
  );

  // Initialize and update D3 visualization
  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    // Nigeria bounds (approximately)
    const bounds = [
      [2, 4], // Southwest corner
      [15, 14.5], // Northeast corner (long, lat)
    ];
    const center = [8.5, 9.3]; // Center of Nigeria

    // Create projection
    const projection = geoMercator()
      .center(center as [number, number])
      .fitSize([width, height], {
        type: 'FeatureCollection',
        features: [
          {
            type: 'Feature',
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [bounds[0][0], bounds[0][1]],
                  [bounds[1][0], bounds[0][1]],
                  [bounds[1][0], bounds[1][1]],
                  [bounds[0][0], bounds[1][1]],
                  [bounds[0][0], bounds[0][1]],
                ],
              ],
            },
          },
        ],
      } as any);

    const path = geoPath().projection(projection);

    // Select SVG
    const svg = select(svgRef.current);
    svg.attr('width', width).attr('height', height);

    // Clear previous content
    svg.selectAll('*').remove();

    // Add background
    svg
      .append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', COLORS.background);

    // Add grid
    const gridGroup = svg.append('g').attr('class', 'grid').attr('opacity', 0.05);
    for (let x of range(bounds[0][0], bounds[1][0], 1)) {
      gridGroup
        .append('line')
        .attr('x1', projection([x, bounds[0][1]])![0])
        .attr('y1', projection([x, bounds[0][1]])![1])
        .attr('x2', projection([x, bounds[1][1]])![0])
        .attr('y2', projection([x, bounds[1][1]])![1])
        .attr('stroke', COLORS.textSecondary)
        .attr('stroke-width', 0.5);
    }
    for (let y of range(bounds[0][1], bounds[1][1], 1)) {
      gridGroup
        .append('line')
        .attr('x1', projection([bounds[0][0], y])![0])
        .attr('y1', projection([bounds[0][0], y])![1])
        .attr('x2', projection([bounds[1][0], y])![0])
        .attr('y2', projection([bounds[1][0], y])![1])
        .attr('stroke', COLORS.textSecondary)
        .attr('stroke-width', 0.5);
    }

    // Add country/state boundaries (simplified)
    const stateGroup = svg.append('g').attr('class', 'states');
    stateGroup
      .append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', width)
      .attr('height', height)
      .attr('fill', 'none')
      .attr('stroke', COLORS.gridLight)
      .attr('stroke-width', 2);

    // Add flow lines
    const flowGroup = svg.append('g').attr('class', 'flows');
    (flowGroup
      .selectAll('line')
      .data(flowData)
      .join('line') as any)
      .attr('x1', (d: any) => projection([d.source.longitude, d.source.latitude])![0])
      .attr('y1', (d: any) => projection([d.source.longitude, d.source.latitude])![1])
      .attr('x2', (d: any) => projection([d.target.longitude, d.target.latitude])![0])
      .attr('y2', (d: any) => projection([d.target.longitude, d.target.latitude])![1])
      .attr('stroke', (d: any) => {
        const color = NODE_TYPE_COLORS[d.source.node_type.toLowerCase()] || COLORS.textSecondary;
        return color;
      })
      .attr('stroke-width', (d: any) => Math.max(1, Math.log(d.volume / 10000 + 1) * 1.5))
      .attr('opacity', 0.3)
      .attr('stroke-linecap', 'round')
      .attr('class', (d: any) => `flow-line flow-${d.id}`);

    // Add flow direction arrows (animated dashes)
    const defs = svg.append('defs');
    defs
      .append('marker')
      .attr('id', 'arrowhead')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto')
      .append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', COLORS.accent);

    // Add nodes
    const nodeGroup = svg
      .append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(visibleNodes)
      .join((enter: any) =>
        enter
          .append('g')
          .attr('class', (d: Node) => `node node-${d.id}`)
          .attr(
            'transform',
            (d: Node) => `translate(${projection([d.longitude, d.latitude])![0]},${projection([d.longitude, d.latitude])![1]})`
          )
          .call((g: any) => {
            // Background ring (pulse effect when not selected)
            g.append('circle')
              .attr('class', 'node-ring')
              .attr('r', (d: Node) => getNodeRadius(d) * 3)
              .attr('fill', 'none')
              .attr('stroke', (d: Node) => getNodeColor(d))
              .attr('stroke-width', 1)
              .attr('opacity', 0.1);

            // Main node circle
            g.append('circle')
              .attr('class', 'node-core')
              .attr('r', (d: Node) => getNodeRadius(d))
              .attr('fill', (d: Node) => getNodeColor(d))
              .attr('opacity', 0.85);

            // Glow shadow
            g.append('circle')
              .attr('class', 'node-glow')
              .attr('r', (d: Node) => getNodeRadius(d))
              .attr('fill', 'none')
              .attr('stroke', (d: Node) => getNodeColor(d))
              .attr('stroke-width', 2)
              .attr('opacity', 0.0)
              .attr('filter', `drop-shadow(0 0 8px ${COLORS.accent})`);

            // Label
            if (visibleNodes.length < 50) {
              g.append('text')
                .attr('class', 'node-label')
                .attr('x', 0)
                .attr('y', (d: Node) => getNodeRadius(d) + 14)
                .attr('text-anchor', 'middle')
                .attr('font-family', "'JetBrains Mono', monospace")
                .attr('font-size', 8)
                .attr('font-weight', 500)
                .attr('fill', COLORS.textSecondary)
                .attr('pointer-events', 'none')
                .text((d: Node) => d.name.substring(0, 12));
            }

            // Status indicator
            g.append('circle')
              .attr('class', 'status-dot')
              .attr('cx', (d: Node) => getNodeRadius(d) + 3)
              .attr('cy', 0)
              .attr('r', 3)
              .attr('fill', (d: Node) => {
                if (d.status === 'operational') return COLORS.success;
                if (d.status === 'maintenance') return COLORS.warning;
                if (d.status === 'shutdown') return COLORS.danger;
                return COLORS.textMuted;
              });
          })
      );

    // Add hover and click interactions
    nodeGroup.on('mouseenter', function (this: SVGGElement, event: MouseEvent, d: Node) {
      select(this)
        .select('.node-core')
        .transition()
        .duration(200)
        .attr('r', getNodeRadius(d) * 1.4);

      select(this)
        .select('.node-glow')
        .attr('opacity', 0.5)
        .transition()
        .duration(200)
        .attr('opacity', 0.8);

      setTooltip({
        node: d,
        x: event.pageX,
        y: event.pageY,
      });
      set_hovered_node(d.id);
    });

    nodeGroup.on('mouseleave', function (this: SVGGElement, event: MouseEvent, d: Node) {
      select(this)
        .select('.node-core')
        .transition()
        .duration(200)
        .attr('r', getNodeRadius(d));

      select(this).select('.node-glow').attr('opacity', 0);

      setTooltip(null);
      set_hovered_node(null);
    });

    nodeGroup.on('click', (event: MouseEvent, d: Node) => {
      event.stopPropagation();
      set_selected_node(d.id);
    });

    // Pan/zoom with mouse
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const delta = event.deltaY > 0 ? 0.9 : 1.1;
      setZoom((z) => Math.max(0.5, Math.min(5, z * delta)));
    };

    const handleMouseDown = (event: MouseEvent) => {
      if ((event.target as HTMLElement).tagName.toLowerCase() === 'text') return;
      setPanning(true);
      setStartPan({ x: event.clientX - translate.x, y: event.clientY - translate.y });
    };

    const handleMouseMove = (event: MouseEvent) => {
      if (!panning) return;
      setTranslate({
        x: event.clientX - startPan.x,
        y: event.clientY - startPan.y,
      });
    };

    const handleMouseUp = () => {
      setPanning(false);
    };

    svg.on('wheel', handleWheel);
    svg.on('mousedown', handleMouseDown);
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [visibleNodes, flowData, getNodeColor, getNodeRadius, selected_node, set_selected_node, set_hovered_node, panning, startPan, translate, zoom]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full overflow-hidden"
      style={{ background: COLORS.background }}
    >
      <svg
        ref={svgRef}
        className="absolute inset-0"
        style={{ cursor: panning ? 'grabbing' : 'grab' }}
      />

      {/* Tooltip */}
      {tooltip && (
        <div
          className="fixed z-50 pointer-events-none"
          style={{
            left: tooltip.x + 14,
            top: tooltip.y - 10,
          }}
        >
          <div
            className="rounded px-3 py-2 text-xs font-mono shadow-xl"
            style={{
              background: 'rgba(10, 14, 39, 0.96)',
              border: '1px solid rgba(14, 165, 233, 0.4)',
              backdropFilter: 'blur(10px)',
              minWidth: '200px',
              boxShadow: '0 0 20px rgba(14, 165, 233, 0.2)',
            }}
          >
            <div className="text-[#0ea5e9] font-bold truncate">{tooltip.node.name}</div>
            <div className="text-[#6b7a96] text-[10px] mt-1 capitalize">
              {tooltip.node.node_type.replace(/_/g, ' ')}
            </div>
            <div className="flex items-center gap-2 mt-2">
              <div
                className="w-2 h-2 rounded-full"
                style={{
                  background:
                    tooltip.node.status === 'operational'
                      ? COLORS.success
                      : tooltip.node.status === 'maintenance'
                        ? COLORS.warning
                        : COLORS.danger,
                }}
              />
              <span className="text-[#b0c0dd] text-[10px] capitalize">
                {tooltip.node.status}
              </span>
            </div>
            {tooltip.node.capacity_bpd && (
              <div className="text-[#0ea5e9] text-[10px] mt-2 font-mono">
                Capacity: {(tooltip.node.capacity_bpd / 1000).toFixed(1)}K BOPD
              </div>
            )}
            {tooltip.node.state && (
              <div className="text-[#6b7a96] text-[9px] mt-1">
                📍 {tooltip.node.state}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Map info overlay */}
      <div
        className="absolute bottom-6 left-6 text-[10px] font-mono pointer-events-none z-10"
        style={{
          color: COLORS.textSecondary,
          textShadow: '0 0 10px rgba(14, 165, 233, 0.2)',
        }}
      >
        <div style={{ color: COLORS.accent }}>
          📡 Spatial Intelligence Network
        </div>
        <div className="mt-1" style={{ color: COLORS.textMuted }}>
          {visibleNodes.length} nodes · {flowData.length} flows active
        </div>
      </div>

      {/* Loading state */}
      {visibleNodes.length === 0 && (
        <div
          className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none"
          style={{ zIndex: 400 }}
        >
          <div className="text-center">
            <div className="text-[#0ea5e9] text-sm font-mono mb-2 font-bold">
              ⏳ INITIALIZING SPATIAL CORE
            </div>
            <div className="text-[#6b7a96] text-xs font-mono">
              Loading Nigeria petroleum network topology...
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

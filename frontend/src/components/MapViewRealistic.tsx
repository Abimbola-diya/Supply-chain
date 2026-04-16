'use client';

import React, { useEffect, useRef, useState } from 'react';

interface MapViewProps {
  nodes?: any[];
  flows?: any[];
}

export default function MapViewRealistic({ nodes = [], flows = [] }: MapViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [osmImage, setOsmImage] = useState<string>('');
  const [nodeSummary, setNodeSummary] = useState<any>(null);
  const [ecosystemNodes, setEcosystemNodes] = useState<Set<string>>(new Set());
  const svgRef = useRef<any>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    
    console.log('🗺️  MapViewRealistic rendering with', nodes.length, 'nodes and', flows?.length || 0, 'flows');
    if (nodes.length > 0) {
      console.log('📍 Sample node:', nodes[0]);
    }

    Promise.all([
      import('d3'),
      import('topojson-client'),
    ]).then(async ([d3Module, topojsonModule]) => {
      const d3 = (d3Module as any).default || d3Module;
      const topojson = (topojsonModule as any).default || topojsonModule;

      const width = containerRef.current!.clientWidth;
      const height = containerRef.current!.clientHeight;

      // SVG
      const svg = d3
        .select(containerRef.current)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .style('display', 'block');

      // Main group for zoom
      const g = svg.append('g');

      // Background
      g.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', '#04080F');

      // Projection centered on Nigeria
      const projection = d3
        .geoMercator()
        .center([8.5, 9.5])
        .scale(4000)
        .translate([width / 2, height / 2]);

      const pathGen = d3.geoPath().projection(projection);

      try {
        // Load world map
        const world = await d3.json(
          'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json'
        );

        const countries = topojson.feature(world, (world as any).objects.countries);
        g.selectAll('.country')
          .data((countries as any).features)
          .enter()
          .append('path')
          .attr('class', 'country')
          .attr('d', pathGen as any)
          .attr('fill', '#040810')
          .attr('stroke', '#0a1420')
          .attr('stroke-width', 0.3);
      } catch (e) {
        console.log('World map load issue');
      }

      try {
        // Load Nigeria TopoJSON
        const nigeria = await d3.json(
          'https://cdn.jsdelivr.net/npm/datamaps@0.5.10/src/js/data/nga.topo.json'
        );

        const statesKey = Object.keys((nigeria as any).objects)[0];
        const states = topojson.feature(nigeria, (nigeria as any).objects[statesKey]);

        // Create tooltip element
        const tooltip = document.createElement('div');
        tooltip.style.cssText = `
          position: absolute;
          padding: 8px 12px;
          background: rgba(10, 24, 40, 0.95);
          border: 1px solid #00d9ff;
          color: #00d9ff;
          border-radius: 3px;
          font-size: 12px;
          font-family: monospace;
          pointer-events: none;
          z-index: 100;
          display: none;
          font-weight: 600;
          letter-spacing: 0.5px;
        `;
        containerRef.current?.appendChild(tooltip);

        // Draw state boundaries
        g.selectAll('.state')
          .data((states as any).features)
          .enter()
          .append('path')
          .attr('class', 'state')
          .attr('d', pathGen as any)
          .attr('fill', '#0a1828')
          .attr('stroke', '#1a4460')
          .attr('stroke-width', 0.8)
          .style('cursor', 'pointer')
          .on('mouseover', function (this: SVGPathElement, event: any, d: any) {
            d3.select(this)
              .attr('fill', '#164070')
              .attr('stroke-width', 1.2);
            
            const stateName = d.properties?.name || 'State';
            tooltip.innerHTML = stateName;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY - 28) + 'px';
          })
          .on('mousemove', function (this: SVGPathElement, event: any) {
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY - 28) + 'px';
          })
          .on('mouseout', function (this: SVGPathElement) {
            d3.select(this)
              .attr('fill', '#0a1828')
              .attr('stroke-width', 0.8);
            tooltip.style.display = 'none';
          })
          .on('click', function (event: any, d: any) {
            event.stopPropagation();
            const stateName = d.properties?.name || 'State';
            tooltip.innerHTML = stateName;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY - 28) + 'px';
          });

        // Draw state borders/mesh
        const mesh = topojson.mesh(
          nigeria,
          (nigeria as any).objects[statesKey],
          (a: any, b: any) => a !== b
        );
        g.append('path')
          .attr('class', 'borders')
          .attr('d', pathGen(mesh) as any)
          .attr('fill', 'none')
          .attr('stroke', '#1a4460')
          .attr('stroke-width', 0.4);

        // Create node lookup map
        const nodeMap = new Map();
        nodes.forEach((n: any) => {
          nodeMap.set(n.id, n);
          nodeMap.set(n.short_id, n);
        });

        // Draw animated flows between nodes
        if (flows && flows.length > 0) {
          const validFlows = flows.filter((f: any) => {
            const source = nodeMap.get(f.source_node_id) || nodeMap.get(f.from_short_id);
            const dest = nodeMap.get(f.destination_node_id) || nodeMap.get(f.to_short_id);
            return source && dest && source.latitude && source.longitude && dest.latitude && dest.longitude;
          });

          console.log('🔀 Drawing', validFlows.length, 'flows');

          // Draw flow paths
          const flowPaths = g.selectAll('.flow-path')
            .data(validFlows)
            .enter()
            .append('path')
            .attr('class', (f: any, i: number) => `flow-path flow-${i}`)
            .attr('d', (f: any) => {
              const source = nodeMap.get(f.source_node_id) || nodeMap.get(f.from_short_id);
              const dest = nodeMap.get(f.destination_node_id) || nodeMap.get(f.to_short_id);
              const [sx, sy] = projection([source.longitude, source.latitude]);
              const [dx, dy] = projection([dest.longitude, dest.latitude]);
              
              // Create curved path
              const mx = (sx + dx) / 2;
              const my = (sy + dy) / 2;
              return `M${sx},${sy}Q${mx},${my} ${dx},${dy}`;
            })
            .attr('fill', 'none')
            .attr('stroke', (f: any) => {
              if (f.is_active) return 'rgba(0, 217, 255, 0.4)';
              return 'rgba(200, 100, 100, 0.2)';
            })
            .attr('stroke-width', (f: any) => {
              const volume = f.avg_volume_value || 0;
              return Math.max(0.5, Math.min(3, volume / 1000));
            })
            .attr('stroke-dasharray', (d: any, i: number) => {
              const length = 20;
              return `${length} ${length}`;
            })
            .attr('stroke-linecap', 'round')
            .style('pointer-events', 'none');
          
          // Store reference for later updating ecosystem flows
          svgRef.current = { g, flowPaths, validFlows, projection, nodeMap };
          
          // Function to update flow styles based on ecosystem
          const updateEcosystemFlows = (ecosystemIds: Set<string>) => {
            if (!svgRef.current) return;
            const { g, validFlows } = svgRef.current;
            
            g.selectAll('.flow-path')
              .attr('stroke', (f: any) => {
                const isInEcosystem = 
                  ecosystemIds.has(f.source_node_id) ||
                  ecosystemIds.has(f.from_short_id) ||
                  ecosystemIds.has(f.destination_node_id) ||
                  ecosystemIds.has(f.to_short_id);
                
                if (isInEcosystem) {
                  return '#00ff88';  // Bright green for ecosystem flows
                } else if (f.is_active) {
                  return 'rgba(0, 217, 255, 0.2)';  // Dim the other flows
                }
                return 'rgba(100, 50, 50, 0.1)';
              })
              .attr('stroke-width', (f: any) => {
                const isInEcosystem = 
                  ecosystemIds.has(f.source_node_id) ||
                  ecosystemIds.has(f.from_short_id) ||
                  ecosystemIds.has(f.destination_node_id) ||
                  ecosystemIds.has(f.to_short_id);
                
                if (isInEcosystem) {
                  const volume = f.avg_volume_value || 0;
                  return Math.max(2, Math.min(5, volume / 500));  // Thicker for ecosystem
                }
                
                const volume = f.avg_volume_value || 0;
                return Math.max(0.3, Math.min(2, volume / 1500));  // Thinner for others
              })
              .attr('opacity', (f: any) => {
                const isInEcosystem = 
                  ecosystemIds.has(f.source_node_id) ||
                  ecosystemIds.has(f.from_short_id) ||
                  ecosystemIds.has(f.destination_node_id) ||
                  ecosystemIds.has(f.to_short_id);
                
                return isInEcosystem ? 1 : 0.15;
              });
          };
          
          // Store the update function for use in cleanup
          svgRef.current.updateEcosystemFlows = updateEcosystemFlows;

          // Animate the dashed lines
          const animate = () => {
            g.selectAll('.flow-path')
              .attr('stroke-dashoffset', (d: any, i: number) => {
                return (Date.now() / 30 + i * 5) % 40;
              });
            requestAnimationFrame(animate);
          };
          animate();
        }

        // Draw petroleum network nodes on the map
        if (nodes && nodes.length > 0) {
          const nodesWithCoords = nodes.filter((n: any) => n.latitude && n.longitude);
          console.log('🟠 Rendering', nodesWithCoords.length, 'nodes with coordinates');
          
          const nodeGroup = g.selectAll('.node')
            .data(nodesWithCoords)
            .enter()
            .append('circle')
            .attr('class', 'node')
            .attr('cx', (d: any) => {
              const projected = projection([d.longitude, d.latitude]);
              return projected ? projected[0] : -1000;
            })
            .attr('cy', (d: any) => {
              const projected = projection([d.longitude, d.latitude]);
              return projected ? projected[1] : -1000;
            })
            .attr('r', 6)
            .attr('fill', (d: any) => {
              const type = (d.node_type || d.type || '').toLowerCase();
              if (type.includes('upstream')) return '#ff6b35';
              if (type.includes('refinery')) return '#c77dff';
              if (type.includes('terminal') || type.includes('export')) return '#ec4899';
              if (type.includes('depot') || type.includes('distributor')) return '#fbbf24';
              if (type.includes('retail')) return '#10b981';
              return '#0ea5e9';
            })
            .attr('opacity', 0.9)
            .attr('stroke', '#fff')
            .attr('stroke-width', 0.5)
            .style('cursor', 'pointer')
            .on('mouseover', function(this: SVGCircleElement, event: any, d: any) {
              d3.select(this)
                .attr('r', 10)
                .attr('opacity', 1);
            })
            .on('mouseout', function(this: SVGCircleElement) {
              d3.select(this)
                .attr('r', 6)
                .attr('opacity', 0.9);
            })
            .on('click', function(event: any, d: any) {
              event.stopPropagation();
              console.log('🔍 Node clicked:', d);
              
              // Update selected node
              setSelectedNode(d);
              
              // Get related flows (supply chain)
              const relatedFlows = (flows || []).filter((f: any) =>
                f.source_node_id === d.id || f.from_short_id === d.id ||
                f.source_node_id === d.short_id || f.from_short_id === d.short_id ||
                f.destination_node_id === d.id || f.to_short_id === d.id ||
                f.destination_node_id === d.short_id || f.to_short_id === d.short_id
              );
              
              // Collect all connected node IDs to build the ecosystem
              const connectedIds = new Set<string>();
              connectedIds.add(d.id || d.short_id);
              
              relatedFlows.forEach((f: any) => {
                if (f.source_node_id) connectedIds.add(f.source_node_id);
                if (f.from_short_id) connectedIds.add(f.from_short_id);
                if (f.destination_node_id) connectedIds.add(f.destination_node_id);
                if (f.to_short_id) connectedIds.add(f.to_short_id);
              });
              
              setEcosystemNodes(connectedIds);
              console.log('🌐 Ecosystem nodes:', connectedIds.size, relatedFlows.length, 'flows');
              
              // Prepare node summary
              setNodeSummary({
                ...d,
                relatedFlows: relatedFlows.length,
                inbound: relatedFlows.filter((f: any) => 
                  f.destination_node_id === d.id || f.to_short_id === d.id ||
                  f.destination_node_id === d.short_id || f.to_short_id === d.short_id
                ).length,
                outbound: relatedFlows.filter((f: any) =>
                  f.source_node_id === d.id || f.from_short_id === d.id ||
                  f.source_node_id === d.short_id || f.from_short_id === d.short_id
                ).length
              });
              
              // Fetch OSM static image
              if (d.latitude && d.longitude) {
                const osmStaticUrl = `https://tile.openstreetmap.org/${Math.floor(21 / 2.5)}/${Math.floor((d.longitude + 180) / 360 * Math.pow(2, Math.floor(21 / 2.5)))}/${Math.floor((1 - Math.log(Math.tan(d.latitude * Math.PI / 180) + 1 / Math.cos(d.latitude * Math.PI / 180)) / Math.PI) / 2 * Math.pow(2, Math.floor(21 / 2.5)))}.png`;
                // Use Mapbox or Stadiamaps static image API
                const mapboxUrl = `https://api.mapbox.com/styles/v1/mapbox/dark-v11/static/${d.longitude},${d.latitude},10,0/600x400@2x?access_token=pk.eyJ1IjoiZXhhbXBsZSIsImEiOiJjazBxMzBhMDAwIn0`;
                const stadiaUrl = `https://tiles.stadiamaps.com/tiles/osm_bright/10/${Math.floor((d.longitude + 180) / 360 * Math.pow(2, 10))}/${Math.floor((1 - Math.log(Math.tan(d.latitude * Math.PI / 180) + 1 / Math.cos(d.latitude * Math.PI / 180)) / Math.PI) / 2 * Math.pow(2, 10))}.png`;
                
                // Use a simple static map URL with OpenStreetMap
                const imageUrl = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/10/${Math.floor((d.longitude + 180) / 360 * Math.pow(2, 10))}/${Math.floor((1 - Math.log(Math.tan(d.latitude * Math.PI / 180) + 1 / Math.cos(d.latitude * Math.PI / 180)) / Math.PI) / 2 * Math.pow(2, 10))}.jpg`;
                setOsmImage(imageUrl);
              }
              
              // Zoom to node
              const [cx, cy] = projection([d.longitude, d.latitude]);
              svg
                .transition()
                .duration(750)
                .call(
                  zoom.transform as any,
                  d3.zoomIdentity
                    .translate(width / 2, height / 2)
                    .scale(6)
                    .translate(-cx, -cy)
                );
            })
            .append('title')
            .text((d: any) => `${d.name || 'Node'} (${d.node_type || d.type || 'Unknown'})`);
        } else {
          console.log('ℹ️ No nodes to render or no nodes with coordinates');
        }
      } catch (e) {
        console.log('Nigeria map load issue:', e);
      }

      // Zoom behavior
      const zoom = d3
        .zoom()
        .scaleExtent([1, 12])
        .on('zoom', (event: any) => {
          g.attr('transform', event.transform);
        });

      svg.call(zoom as any);

      // Reset button
      const resetBtn = document.createElement('button');
      resetBtn.innerHTML = '↺ Reset';
      resetBtn.style.cssText = `
        position: absolute;
        top: 60px;
        right: 20px;
        padding: 8px 14px;
        background: rgba(10,24,40,0.9);
        border: 1px solid #1a4460;
        color: #00d9ff;
        cursor: pointer;
        font-size: 12px;
        font-family: monospace;
        border-radius: 3px;
        z-index: 10;
        transition: all 0.2s;
      `;
      resetBtn.onmouseover = () => {
        resetBtn.style.background = 'rgba(20,50,80,0.95)';
        resetBtn.style.color = '#00ffff';
      };
      resetBtn.onmouseout = () => {
        resetBtn.style.background = 'rgba(10,24,40,0.9)';
        resetBtn.style.color = '#00d9ff';
      };
      resetBtn.onclick = () => {
        svg
          .transition()
          .duration(750)
          .call(
            zoom.transform as any,
            d3.zoomIdentity.translate(width / 2, height / 2)
          );
      };
        containerRef.current?.appendChild(resetBtn);

      // Info text
      const infoDiv = document.createElement('div');
      infoDiv.innerHTML = 'Scroll to zoom · Drag to pan';
      infoDiv.style.cssText = `
        position: absolute;
        bottom: 20px;
        left: 20px;
        font-size: 11px;
        color: #546880;
        font-family: monospace;
        z-index: 10;
      `;
      containerRef.current?.appendChild(infoDiv);

      // Cleanup resize
      const handleResize = () => {
        const nw = containerRef.current?.clientWidth || width;
        const nh = containerRef.current?.clientHeight || height;
        svg.attr('width', nw).attr('height', nh);
        projection.translate([nw / 2, nh / 2]);
      };

      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
    });

    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = '';
      }
    };
  }, [nodes, flows]);

  // Effect to update ecosystem highlighting
  useEffect(() => {
    if (svgRef.current && svgRef.current.updateEcosystemFlows) {
      if (ecosystemNodes.size > 0) {
        // Update flows
        svgRef.current.updateEcosystemFlows(ecosystemNodes);
        
        // Update nodes
        if (svgRef.current.g) {
          svgRef.current.g.selectAll('circle.node')
            .attr('r', (d: any) => {
              const isInEcosystem = ecosystemNodes.has(d.id) || ecosystemNodes.has(d.short_id);
              return isInEcosystem ? 9 : 6;
            })
            .attr('opacity', (d: any) => {
              const isInEcosystem = ecosystemNodes.has(d.id) || ecosystemNodes.has(d.short_id);
              return isInEcosystem ? 1 : 0.4;
            })
            .attr('filter', (d: any) => {
              if (selectedNode && (d.id === selectedNode.id || d.short_id === selectedNode.short_id)) {
                return 'drop-shadow(0 0 8px #00ff88)';
              }
              return 'none';
            });
        }
      } else {
        // Reset all flows and nodes
        if (svgRef.current.g) {
          svgRef.current.g.selectAll('.flow-path')
            .attr('stroke', (f: any) => {
              if (f.is_active) return 'rgba(0, 217, 255, 0.4)';
              return 'rgba(200, 100, 100, 0.2)';
            })
            .attr('stroke-width', (f: any) => {
              const volume = f.avg_volume_value || 0;
              return Math.max(0.5, Math.min(3, volume / 1000));
            })
            .attr('opacity', 0.8);
          
          svgRef.current.g.selectAll('circle.node')
            .attr('r', 6)
            .attr('opacity', 0.9)
            .attr('filter', 'none');
        }
      }
    }
  }, [ecosystemNodes, selectedNode]);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        background: '#04080F',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Detail Panel */}
      {selectedNode && (
        <div
          style={{
            position: 'absolute',
            top: 60,
            right: 20,
            width: 380,
            maxHeight: 'calc(100vh - 100px)',
            background: 'rgba(10, 24, 40, 0.95)',
            border: '1px solid #1a4460',
            borderRadius: 8,
            padding: 20,
            zIndex: 50,
            overflow: 'auto',
            color: '#d0d8e0',
            fontSize: 12,
            fontFamily: 'monospace',
            boxShadow: '0 8px 32px rgba(0, 217, 255, 0.1)',
          }}
        >
          {/* Close button */}
          <button
            onClick={() => {
              setSelectedNode(null);
              setOsmImage('');
              setNodeSummary(null);
              setEcosystemNodes(new Set());
            }}
            style={{
              position: 'absolute',
              top: 10,
              right: 10,
              background: 'transparent',
              border: 'none',
              color: '#00d9ff',
              cursor: 'pointer',
              fontSize: 18,
            }}
          >
            ✕
          </button>

          {/* Node name and type */}
          <h2 style={{ margin: '0 0 15px 0', color: '#00d9ff', fontSize: 16 }}>
            {selectedNode.name || 'Node'}
          </h2>

          {/* Type badge */}
          <div style={{ marginBottom: 15 }}>
            <span
              style={{
                display: 'inline-block',
                background: 'rgba(0, 217, 255, 0.1)',
                border: '1px solid #00d9ff',
                color: '#00d9ff',
                padding: '4px 8px',
                borderRadius: 3,
                fontSize: 10,
                fontWeight: 600,
                textTransform: 'uppercase',
              }}
            >
              {(selectedNode.node_type || selectedNode.type || 'Unknown').replace(/_/g, ' ')}
            </span>
          </div>

          {/* Location info */}
          <div style={{ marginBottom: 15, paddingBottom: 15, borderBottom: '1px solid #1a4460' }}>
            <div style={{ marginBottom: 8 }}>
              <span style={{ color: '#8899aa' }}>LOCATION</span>
            </div>
            <div style={{ fontSize: 11, lineHeight: 1.6 }}>
              <div>State: <span style={{ color: '#00d9ff' }}>{selectedNode.state || 'N/A'}</span></div>
              <div>Zone: <span style={{ color: '#00d9ff' }}>{selectedNode.geopolitical_zone || 'N/A'}</span></div>
              <div>Lat: <span style={{ color: '#00d9ff' }}>{selectedNode.latitude?.toFixed(4)}</span></div>
              <div>Lng: <span style={{ color: '#00d9ff' }}>{selectedNode.longitude?.toFixed(4)}</span></div>
            </div>
          </div>

          {/* Supply Chain Info */}
          {nodeSummary && (
            <div style={{ marginBottom: 15, paddingBottom: 15, borderBottom: '1px solid #1a4460' }}>
              <div style={{ marginBottom: 8 }}>
                <span style={{ color: '#8899aa' }}>SUPPLY CHAIN</span>
              </div>
              <div style={{ fontSize: 11, lineHeight: 2 }}>
                <div>→ Inbound: <span style={{ color: '#fbbf24' }}>{nodeSummary.inbound || 0}</span> flows</div>
                <div>← Outbound: <span style={{ color: '#10b981' }}>{nodeSummary.outbound || 0}</span> flows</div>
                <div>Total Connected: <span style={{ color: '#00d9ff' }}>{nodeSummary.relatedFlows || 0}</span></div>
              </div>
            </div>
          )}

          {/* Additional Info */}
          <div style={{ marginBottom: 15, paddingBottom: 15, borderBottom: '1px solid #1a4460' }}>
            <div style={{ marginBottom: 8 }}>
              <span style={{ color: '#8899aa' }}>STATUS</span>
            </div>
            <div style={{ fontSize: 11, lineHeight: 2 }}>
              <div>
                Status:{' '}
                <span
                  style={{
                    color: selectedNode.status === 'operational' ? '#10b981' : '#ff6b35',
                  }}
                >
                  {selectedNode.status || 'Unknown'}
                </span>
              </div>
              <div>
                Active:{' '}
                <span style={{ color: selectedNode.is_active ? '#10b981' : '#ff6b35' }}>
                  {selectedNode.is_active ? 'Yes' : 'No'}
                </span>
              </div>
              <div>
                Confidence:{' '}
                <span style={{ color: '#00d9ff' }}>{selectedNode.confidence_level || 'N/A'}</span>
              </div>
            </div>
          </div>

          {/* OSM Map Image */}
          {osmImage && (
            <div style={{ marginBottom: 15 }}>
              <div style={{ marginBottom: 8 }}>
                <span style={{ color: '#8899aa' }}>LOCATION MAP</span>
              </div>
              <img
                src={osmImage}
                alt="Location map"
                style={{
                  width: '100%',
                  borderRadius: 4,
                  maxHeight: 200,
                  objectFit: 'cover',
                  border: '1px solid #1a4460',
                }}
                onError={(e: any) => {
                  e.target.style.display = 'none';
                }}
              />
            </div>
          )}

          {/* Raw details */}
          <div style={{ fontSize: 10, color: '#8899aa' }}>
            <div>ID: <span style={{ color: '#546880' }}>{selectedNode.short_id || selectedNode.id}</span></div>
            <div>Updated: <span style={{ color: '#546880' }}>{new Date(selectedNode.data_as_of || Date.now()).toLocaleDateString()}</span></div>
          </div>
        </div>
      )}
    </div>
  );
}

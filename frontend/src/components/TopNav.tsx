'use client';

import React, { useState, useCallback } from 'react';
import { Map as MapIcon, Network, GitBranch, MessageSquare, Search, X, ChevronRight } from 'lucide-react';
import { useStore } from '@/stores';
import { ViewMode, NODE_TYPE_CONFIG } from '@/types';
import { searchNodes } from '@/lib/supabase';
import type { Node } from '@/types';

const VIEWS: { id: ViewMode; label: string; icon: React.ReactNode; shortLabel: string }[] = [
  { id: 'map',       label: 'End-to-End Map',  shortLabel: 'MAP',       icon: <MapIcon size={12} /> },
  { id: 'ecosystem', label: 'Ecosystem',        shortLabel: 'ECOSYSTEM', icon: <Network size={12} /> },
  { id: 'sankey',    label: 'Sankey Flow',      shortLabel: 'SANKEY',    icon: <GitBranch size={12} /> },
];

export default function TopNav() {
  const {
    active_view, set_view,
    toggle_chat, chat_open,
    nodes, flows,
    camera,
    set_camera, set_selected_node,
  } = useStore();

  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Node[]>([]);
  const [searching, setSearching] = useState(false);

  const handleSearch = useCallback(async (q: string) => {
    setSearchQuery(q);
    if (q.length < 2) { setSearchResults([]); return; }
    setSearching(true);
    try {
      const { nodes: results } = await searchNodes(q);

      // Local store search to pick up operator / basin matches
      const localMatches = Array.from(nodes.values()).filter((n) => {
        const lc = q.toLowerCase();
        const nameMatch = n.name && n.name.toLowerCase().includes(lc);
        const stateMatch = n.state && n.state.toLowerCase().includes(lc);
        const op = (n.operator || (n.metadata && (n.metadata as any).operator) || '').toString();
        const basin = (n.basin || (n.metadata && (n.metadata as any).basin) || '').toString();
        const operatorMatch = op && op.toLowerCase().includes(lc);
        const basinMatch = basin && basin.toLowerCase().includes(lc);
        return Boolean(nameMatch || stateMatch || operatorMatch || basinMatch);
      });

      // Merge supabase + local results, dedupe by id
      const map = new Map<string, Node>();
      (results || []).forEach((r) => map.set(r.id, r));
      localMatches.forEach((r) => map.set(r.id, r));
      const merged = Array.from(map.values()).slice(0, 12);
      setSearchResults(merged.slice(0, 8));
    } finally {
      setSearching(false);
    }
  }, []);

  const flyToNode = useCallback((node: Node) => {
    set_camera({ longitude: node.longitude, latitude: node.latitude, zoom: 12 });
    set_selected_node(node.id);
    setSearchOpen(false);
    setSearchQuery('');
    setSearchResults([]);
  }, [set_camera, set_selected_node]);

  const now = new Date();
  const timeStr = now.toUTCString().slice(17, 25) + ' UTC';

  return (
    <div
      className="absolute top-0 left-0 right-0 h-11 z-40 flex items-center px-3 gap-0"
      style={{
        background: 'linear-gradient(180deg, rgba(13,17,23,0.98) 0%, rgba(8,11,30,0.95) 100%)',
        borderBottom: '1px solid rgba(0,217,255,0.15)',
        backdropFilter: 'blur(4px)',
      }}
    >
      {/* Brand */}
      <div className="flex items-center gap-2.5 pr-4 border-r border-[#1A1F3A] mr-3 flex-shrink-0">
        <div className="flex items-center gap-1.5">
          <div
            className="w-6 h-6 rounded flex items-center justify-center flex-shrink-0"
            style={{ background: 'rgba(0,217,255,0.12)', border: '1px solid rgba(0,217,255,0.3)' }}
          >
            <span className="text-[8px] font-bold text-[#00D9FF] tracking-wider">PMS</span>
          </div>
          <div className="hidden sm:block">
            <div className="text-[11px] font-bold text-[#E8E8E8] font-mono tracking-widest leading-none">
              SPATIAL INTELLIGENCE
            </div>
            <div className="text-[8px] text-[#696969] font-mono tracking-widest leading-none mt-0.5">
              NIGERIA CRUDE · v1.0
            </div>
          </div>
        </div>
      </div>

      {/* View switcher */}
      <div className="flex items-center gap-0.5 mr-3 flex-shrink-0">
        {VIEWS.map((v) => {
          const active = active_view === v.id;
          return (
            <button
              key={v.id}
              onClick={() => set_view(v.id)}
              className="flex items-center gap-1.5 px-3 h-7 rounded text-[10px] font-mono font-medium tracking-wider transition-all"
              style={{
                background: active ? 'rgba(0,217,255,0.12)' : 'transparent',
                color: active ? '#00D9FF' : '#555',
                border: active ? '1px solid rgba(0,217,255,0.25)' : '1px solid transparent',
              }}
            >
              {v.icon}
              <span className="hidden lg:block">{v.label}</span>
              <span className="lg:hidden">{v.shortLabel}</span>
            </button>
          );
        })}
      </div>

      {/* Search */}
      <div className="relative flex-shrink-0 mr-3">
        {searchOpen ? (
          <div className="flex items-center gap-1.5">
            <div
              className="flex items-center gap-2 h-7 px-2.5 rounded"
              style={{ background: 'rgba(0,217,255,0.06)', border: '1px solid rgba(0,217,255,0.2)', width: '220px' }}
            >
              <Search size={11} className="text-[#00D9FF] flex-shrink-0" />
              <input
                autoFocus
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                placeholder="Search facilities..."
                className="flex-1 bg-transparent text-[11px] font-mono text-[#E8E8E8] placeholder-[#555] outline-none"
              />
              {searching && <div className="w-2 h-2 rounded-full bg-[#00D9FF] animate-pulse flex-shrink-0" />}
            </div>
            <button onClick={() => { setSearchOpen(false); setSearchQuery(''); setSearchResults([]); }}
              className="text-[#555] hover:text-[#E8E8E8] transition-colors">
              <X size={13} />
            </button>
          </div>
        ) : (
          <button
            onClick={() => setSearchOpen(true)}
            className="flex items-center gap-1.5 h-7 px-2.5 rounded text-[10px] font-mono text-[#555] hover:text-[#00D9FF] transition-colors"
            style={{ border: '1px solid rgba(255,255,255,0.06)' }}
          >
            <Search size={12} />
            <span className="hidden md:block">Search</span>
          </button>
        )}

        {/* Search results dropdown */}
        {searchResults.length > 0 && (
          <div
            className="absolute top-9 left-0 w-[320px] rounded overflow-hidden z-50"
            style={{ background: 'rgba(8,11,30,0.98)', border: '1px solid rgba(0,217,255,0.2)', boxShadow: '0 8px 32px rgba(0,0,0,0.6)' }}
          >
            {searchResults.map((node) => {
              const cfg = NODE_TYPE_CONFIG[node.node_type.toLowerCase()];
              return (
                <button
                  key={node.id}
                  onClick={() => flyToNode(node)}
                  className="w-full flex items-center gap-2.5 px-3 py-2 hover:bg-[#1A1F3A] transition-colors border-b border-[#0D1117] last:border-0"
                >
                  <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: cfg?.color || '#A0A0A0', boxShadow: `0 0 4px ${cfg?.color || '#A0A0A0'}` }} />
                  <div className="flex-1 min-w-0 text-left">
                    <div className="text-[11px] font-mono text-[#E8E8E8] truncate">{node.name}</div>
                    <div className="text-[9px] font-mono text-[#696969]">{cfg?.label || node.node_type} · {node.state || '—'}</div>
                  </div>
                  <ChevronRight size={10} className="text-[#555] flex-shrink-0" />
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Live stats */}
      <div className="hidden md:flex items-center gap-4 text-[9px] font-mono text-[#555] mr-4 flex-shrink-0">
        <div className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-[#00E676] animate-pulse-dot" />
          <span className="text-[#A0A0A0]">{nodes.size.toLocaleString()}</span>
          <span>NODES</span>
        </div>
        <div className="text-[#333]">·</div>
        <div>
          <span className="text-[#A0A0A0]">{flows.size.toLocaleString()}</span>
          <span className="ml-1">FLOWS</span>
        </div>
        <div className="text-[#333]">·</div>
        <div className="hidden lg:block">
          <span className="text-[#A0A0A0]">{camera.longitude.toFixed(3)}</span>
          <span className="mx-0.5">,</span>
          <span className="text-[#A0A0A0]">{camera.latitude.toFixed(3)}</span>
        </div>
        <div className="text-[#333] hidden lg:block">·</div>
        <div className="hidden lg:block">Z{camera.zoom.toFixed(1)}</div>
      </div>

      {/* Jarvis button */}
      <button
        onClick={toggle_chat}
        className="flex items-center gap-1.5 h-7 px-3 rounded text-[10px] font-mono font-medium tracking-wider transition-all flex-shrink-0"
        style={{
          background: chat_open ? 'rgba(157,78,221,0.18)' : 'rgba(157,78,221,0.06)',
          color: chat_open ? '#E0AAFF' : '#9D4EDD',
          border: `1px solid ${chat_open ? 'rgba(157,78,221,0.4)' : 'rgba(157,78,221,0.2)'}`,
        }}
      >
        <MessageSquare size={12} />
        <span>JARVIS</span>
        {chat_open && <span className="w-1.5 h-1.5 rounded-full bg-[#E0AAFF] animate-pulse-dot" />}
      </button>
    </div>
  );
}

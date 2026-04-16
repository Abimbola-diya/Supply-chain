// ============================================================================
// CORE DOMAIN TYPES
// ============================================================================

export type NodeType =
  | 'refinery'
  | 'upstream_field'
  | 'terminal'
  | 'jetty'
  | 'pipeline'
  | 'retail_station'
  | 'distribution_center'
  | 'import_point'
  | 'export_point'
  | 'storage';

export type NodeStatus = 'operational' | 'maintenance' | 'shutdown' | 'degraded' | 'unknown';

export type FlowType =
  | 'crude_supply'
  | 'product_transfer'
  | 'retail_distribution'
  | 'export_route'
  | 'import_route';

export type ProductType = 'crude_oil' | 'pms' | 'ago' | 'kerosene' | 'lpng' | 'bitumen';

export type TransportMode = 'pipeline' | 'vessel' | 'truck' | 'rail';

export type ViewMode = 'map' | 'ecosystem' | 'sankey' | 'hud';

export type AlertLevel = 'healthy' | 'caution' | 'critical';

// ============================================================================
// DATA MODELS
// ============================================================================

export interface Node {
  id: string;
  name: string;
  node_type: string;
  latitude: number;
  longitude: number;
  status: string;
  state?: string;
  lga?: string;
  operator?: string;
  basin?: string;
  capacity_bpd?: number;
  connected_flows?: number;
  last_updated?: string;
  metadata?: Record<string, unknown>;
  alert_level?: AlertLevel;
}

export interface Flow {
  id: string;
  source_node_id: string;
  destination_node_id: string;
  transport_mode: string;
  flow_type: string;
  product_type?: string;
  volume_bpd?: number;
  avg_volume_value?: number;
  route_distance_km?: number;
  estimated_transit_days?: number;
  status: string;
  is_active?: boolean;
  operational_frequency?: string;
  last_updated?: string;
  metadata?: Record<string, unknown>;
}

export interface Incident {
  id: string;
  title: string;
  description: string;
  incident_type: string;
  affected_node_id?: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  start_time: string;
  end_time?: string;
  impact_volume?: number;
  resolution?: string;
}

export interface MacroIndicator {
  id: string;
  indicator_name: string;
  category: string;
  value: number;
  unit: string;
  period_start: string;
  period_end: string;
  trend?: 'up' | 'down' | 'stable';
  yoy_change_pct?: number;
}

export interface FlowMetric {
  timestamp: string;
  node_id?: string;
  flow_id?: string;
  metric_type: string;
  value: number;
  unit: string;
  confidence: number;
}

export interface RagDocument {
  id: string;
  title: string;
  content: string;
  source: string;
  created_at: string;
  metadata?: Record<string, unknown>;
}

export interface InternationalShipment {
  id: string;
  origin_port: string;
  destination_port: string;
  product_type: string;
  quantity: number;
  departure_date: string;
  arrival_date?: string;
  status: string;
}

// ============================================================================
// UI / VISUALIZATION TYPES
// ============================================================================

export interface CameraState {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export type LayerKey =
  | 'flows'
  | 'ais_tracking'
  | 'weather'
  | 'news'
  | 'price_heatmap'
  | 'anomalies'
  | 'pipelines'
  | 'incidents'
  | 'production_heatmap'
  | 'satellite'
  | 'terminals';

export interface NodeTypeConfig {
  label: string;
  color: string;
  hexColor: [number, number, number];
}

// Node type → color mapping (used across map + sidebar)
export const NODE_TYPE_CONFIG: Record<string, NodeTypeConfig> = {
  refinery:            { label: 'Refinery',            color: '#FFD700', hexColor: [255, 215, 0] },
  upstream_field:      { label: 'Upstream Field',      color: '#FFA500', hexColor: [255, 165, 0] },
  terminal:            { label: 'Terminal',             color: '#00D9FF', hexColor: [0, 217, 255] },
  jetty:               { label: 'Jetty',                color: '#0066CC', hexColor: [0, 102, 204] },
  pipeline:            { label: 'Pipeline',             color: '#FFE082', hexColor: [255, 224, 130] },
  retail_station:      { label: 'Retail Station',       color: '#1ABC9C', hexColor: [26, 188, 156] },
  distribution_center: { label: 'Distribution Center', color: '#9D4EDD', hexColor: [157, 78, 221] },
  import_point:        { label: 'Import Point',         color: '#00E676', hexColor: [0, 230, 118] },
  export_point:        { label: 'Export Point',         color: '#FF1744', hexColor: [255, 23, 68] },
  storage:             { label: 'Storage',              color: '#78909C', hexColor: [120, 144, 156] },
};

export const STATUS_COLOR: Record<string, string> = {
  operational: '#00E676',
  maintenance:  '#FFA500',
  shutdown:     '#FF1744',
  degraded:     '#FFD700',
  unknown:      '#78909C',
};

export const FLOW_COLOR: Record<string, [number, number, number]> = {
  crude_oil:        [255, 215, 0],
  pms:              [0, 217, 255],
  ago:              [26, 188, 156],
  kerosene:         [255, 224, 130],
  lpng:             [157, 78, 221],
  bitumen:          [120, 144, 156],
  crude_supply:     [255, 215, 0],
  product_transfer: [0, 217, 255],
  retail_distribution: [26, 188, 156],
  export_route:     [255, 23, 68],
  import_route:     [0, 230, 118],
};

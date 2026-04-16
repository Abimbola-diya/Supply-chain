/**
 * Supabase Client Configuration & Typed Queries
 * Provides type-safe access to all PMS data with real-time subscriptions
 */

import { createClient } from '@supabase/supabase-js';
import {
  Node,
  Flow,
  FlowMetric,
  MacroIndicator,
  Incident,
  RagDocument,
  InternationalShipment,
} from '@/types';

// ============================================================================
// Client Initialization
// ============================================================================

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  console.error(
    '❌ Missing Supabase environment variables. Check .env.local for NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY'
  );
}

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// ============================================================================
// Typed Query Functions
// ============================================================================

/**
 * Fetch all nodes (facilities) from the database
 * Includes geocoding data and status information
 */
export async function fetchAllNodes(): Promise<{ nodes: Node[]; error?: string }> {
  try {
    const { data, error } = await supabase.from('nodes').select(`
      short_id,
      name,
      type,
      latitude,
      longitude,
      status,
      state,
      geopolitical_zone,
      ownership_type,
      is_active,
      confidence_level,
      data_as_of,
      notes
    `);

    if (error) throw error;

    // Map fields to match Node interface
    const mappedNodes = (data || []).map((node: any) => ({
      id: node.short_id,
      short_id: node.short_id,
      name: node.name,
      type: node.type,
      node_type: node.type,
      latitude: node.latitude,
      longitude: node.longitude,
      status: node.status || (node.is_active ? 'operational' : 'shutdown'),
      state: node.state,
      geopolitical_zone: node.geopolitical_zone,
      ownership_type: node.ownership_type,
      is_active: node.is_active,
      confidence_level: node.confidence_level,
      data_as_of: node.data_as_of,
      notes: node.notes
    }));

    return {
      nodes: mappedNodes as Node[],
    };
  } catch (err: any) {
    console.error('❌ Error fetching nodes:', err);
    return {
      nodes: [],
      error: err.message,
    };
  }
}

/**
 * Fetch a single node by ID with all related data
 */
export async function fetchNodeById(nodeId: string): Promise<{ node: Node | null; error?: string }> {
  try {
    const { data, error } = await supabase
      .from('nodes')
      .select()
      .eq('id', nodeId)
      .single();

    if (error) throw error;

    return { node: data as Node };
  } catch (err: any) {
    console.error('❌ Error fetching node:', err);
    return {
      node: null,
      error: err.message,
    };
  }
}

/**
 * Fetch nodes by type (e.g., all refineries)
 */
export async function fetchNodesByType(type: string): Promise<{ nodes: Node[]; error?: string }> {
  try {
    const { data, error } = await supabase
      .from('nodes')
      .select()
      .eq('type', type);

    if (error) throw error;

    return { nodes: data as Node[] };
  } catch (err: any) {
    console.error('❌ Error fetching nodes by type:', err);
    return {
      nodes: [],
      error: err.message,
    };
  }
}

/**
 * Fetch all flows (routes between nodes)
 */
export async function fetchAllFlows(): Promise<{ flows: Flow[]; error?: string }> {
  try {
    const { data, error } = await supabase.from('flows').select(`
      id,
      from_short_id,
      to_short_id,
      transport_mode,
      flow_type,
      avg_volume_value,
      avg_volume_unit,
      is_active,
      is_international,
      confidence_level,
      notes
    `);

    if (error) throw error;

    // Map fields to match Flow interface
    const mappedFlows = (data || []).map((flow: any) => ({
      id: flow.id,
      from_short_id: flow.from_short_id,
      to_short_id: flow.to_short_id,
      source_node_id: flow.from_short_id,
      destination_node_id: flow.to_short_id,
      transport_mode: flow.transport_mode,
      flow_type: flow.flow_type,
      product_type: flow.flow_type,
      avg_volume_value: flow.avg_volume_value,
      avg_volume_unit: flow.avg_volume_unit,
      volume_bpd: flow.avg_volume_value,
      is_active: flow.is_active,
      is_international: flow.is_international,
      status: flow.is_active ? 'operational' : 'shutdown',
      confidence_level: flow.confidence_level,
      notes: flow.notes
    }));

    return {
      flows: mappedFlows as Flow[],
    };
  } catch (err: any) {
    console.error('❌ Error fetching flows:', err);
    return {
      flows: [],
      error: err.message,
    };
  }
}

/**
 * Fetch flows connected to a specific node
 */
export async function fetchFlowsForNode(
  nodeId: string
): Promise<{ inbound: Flow[]; outbound: Flow[]; error?: string }> {
  try {
    const [inboundRes, outboundRes] = await Promise.all([
      supabase.from('flows').select().eq('to_short_id', nodeId),
      supabase.from('flows').select().eq('from_short_id', nodeId),
    ]);

    if (inboundRes.error) throw inboundRes.error;
    if (outboundRes.error) throw outboundRes.error;

    const mapFlow = (flow: any) => ({
      id: flow.id,
      from_short_id: flow.from_short_id,
      to_short_id: flow.to_short_id,
      source_node_id: flow.from_short_id,
      destination_node_id: flow.to_short_id,
      transport_mode: flow.transport_mode,
      flow_type: flow.flow_type,
      avg_volume_value: flow.avg_volume_value,
      avg_volume_unit: flow.avg_volume_unit,
      volume_bpd: flow.avg_volume_value,
      is_active: flow.is_active,
      status: flow.is_active ? 'operational' : 'shutdown'
    });

    return {
      inbound: ((inboundRes.data as any[]) || []).map(mapFlow) as Flow[],
      outbound: ((outboundRes.data as any[]) || []).map(mapFlow) as Flow[],
    };
  } catch (err: any) {
    console.error('❌ Error fetching flows for node:', err);
    return {
      inbound: [],
      outbound: [],
      error: err.message,
    };
  }
}

/**
 * Fetch flow metrics (time-series volume data)
 */
export async function fetchFlowMetrics(
  flowId: string,
  opts?: { startDate?: Date; endDate?: Date }
): Promise<{ metrics: FlowMetric[]; error?: string }> {
  try {
    let query = supabase.from('node_metrics').select().eq('flow_id', flowId);

    if (opts?.startDate) {
      query = query.gte('timestamp', opts.startDate.toISOString());
    }
    if (opts?.endDate) {
      query = query.lte('timestamp', opts.endDate.toISOString());
    }

    const { data, error } = await query.order('timestamp', { ascending: false }).limit(500);

    if (error) throw error;

    return {
      metrics: data as FlowMetric[],
    };
  } catch (err: any) {
    console.error('❌ Error fetching flow metrics:', err);
    return {
      metrics: [],
      error: err.message,
    };
  }
}

/**
 * Fetch macro indicators (industry-level metrics)
 */
export async function fetchMacroIndicators(): Promise<{ indicators: MacroIndicator[]; error?: string }> {
  try {
    const { data, error } = await supabase
      .from('macro_indicators')
      .select()
      .order('timestamp', { ascending: false })
      .limit(100);

    if (error) throw error;

    return {
      indicators: data as MacroIndicator[],
    };
  } catch (err: any) {
    console.error('❌ Error fetching macro indicators:', err);
    return {
      indicators: [],
      error: err.message,
    };
  }
}

/**
 * Fetch recent incidents and events
 */
export async function fetchIncidents(opts?: {
  limit?: number;
  severity?: string;
}): Promise<{ incidents: Incident[]; error?: string }> {
  try {
    let query = supabase
      .from('incidents_and_events')
      .select(`
        title,
        event_type,
        severity,
        affected_node_short_id,
        started_at,
        ended_at,
        is_ongoing,
        impact_metric_name,
        impact_value,
        impact_unit,
        causal_mechanism,
        secondary_effects,
        confidence_level,
        notes
      `)
      .order('started_at', { ascending: false });

    if (opts?.limit) {
      query = query.limit(opts.limit);
    }
    if (opts?.severity) {
      query = query.eq('severity', opts.severity);
    }

    const { data, error } = await query;

    if (error) throw error;

    // Map fields to match Incident interface
    const mappedIncidents = (data || []).map((incident: any, idx: number) => ({
      id: `incident_${idx}`,
      title: incident.title,
      description: incident.notes || incident.causal_mechanism || incident.title || '',
      event_type: incident.event_type,
      incident_type: incident.event_type,
      severity: incident.severity,
      affected_node_short_id: incident.affected_node_short_id,
      affected_node_id: incident.affected_node_short_id,
      started_at: incident.started_at,
      ended_at: incident.ended_at,
      start_time: incident.started_at,
      end_time: incident.ended_at,
      is_ongoing: incident.is_ongoing,
      impact_metric_name: incident.impact_metric_name,
      impact_value: incident.impact_value,
      impact_volume: incident.impact_value,
      impact_unit: incident.impact_unit,
      causal_mechanism: incident.causal_mechanism,
      resolution: incident.causal_mechanism,
      secondary_effects: incident.secondary_effects,
      confidence_level: incident.confidence_level,
      notes: incident.notes
    }));

    return {
      incidents: mappedIncidents as Incident[],
    };
  } catch (err: any) {
    console.error('❌ Error fetching incidents:', err);
    return {
      incidents: [],
      error: err.message,
    };
  }
}

/**
 * Fetch international shipments
 */
export async function fetchInternationalShipments(): Promise<{
  shipments: InternationalShipment[];
  error?: string;
}> {
  try {
    const { data, error } = await supabase
      .from('international_shipments')
      .select()
      .order('shipping_date', { ascending: false })
      .limit(100);

    if (error) throw error;

    return {
      shipments: data as InternationalShipment[],
    };
  } catch (err: any) {
    console.error('❌ Error fetching international shipments:', err);
    return {
      shipments: [],
      error: err.message,
    };
  }
}

/**
 * Fetch RAG documents (regulations, analysis, governance)
 */
export async function fetchRagDocuments(): Promise<{ docs: RagDocument[]; error?: string }> {
  try {
    const { data, error } = await supabase
      .from('rag_documents')
      .select()
      .order('created_at', { ascending: false });

    if (error) throw error;

    return {
      docs: data as RagDocument[],
    };
  } catch (err: any) {
    console.error('❌ Error fetching RAG documents:', err);
    return {
      docs: [],
      error: err.message,
    };
  }
}

/**
 * Search nodes by name or state
 */
export async function searchNodes(query: string): Promise<{ nodes: Node[]; error?: string }> {
  try {
    const { data, error } = await supabase
      .from('nodes')
      .select()
      .or(`name.ilike.%${query}%,state.ilike.%${query}%`)
      .limit(20);

    if (error) throw error;

    return {
      nodes: data as Node[],
    };
  } catch (err: any) {
    console.error('❌ Error searching nodes:', err);
    return {
      nodes: [],
      error: err.message,
    };
  }
}

/**
 * Subscribe to real-time node updates
 */
export function subscribeToNodes(callback: (nodes: Node[]) => void) {
  const subscription = supabase
    .channel('nodes:*')
    .on('postgres_changes', { event: '*', schema: 'public', table: 'nodes' }, () => {
      // Refetch on any change
      fetchAllNodes().then(({ nodes }) => callback(nodes));
    })
    .subscribe();

  return () => {
    subscription.unsubscribe();
  };
}

/**
 * Subscribe to real-time flow updates
 */
export function subscribeToFlows(callback: (flows: Flow[]) => void) {
  const subscription = supabase
    .channel('flows:*')
    .on('postgres_changes', { event: '*', schema: 'public', table: 'flows' }, () => {
      // Refetch on any change
      fetchAllFlows().then(({ flows }) => callback(flows));
    })
    .subscribe();

  return () => {
    subscription.unsubscribe();
  };
}

/**
 * Subscribe to incidents in real-time
 */
export function subscribeToIncidents(callback: (incidents: Incident[]) => void) {
  const subscription = supabase
    .channel('incidents:*')
    .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'incidents_and_events' }, () => {
      // Fetch new incidents
      fetchIncidents({ limit: 50 }).then(({ incidents }) => callback(incidents));
    })
    .subscribe();

  return () => {
    subscription.unsubscribe();
  };
}

/**
 * Update node status (e.g., mark as operational or down)
 */
export async function updateNodeStatus(
  nodeId: string,
  status: string
): Promise<{ success: boolean; error?: string }> {
  try {
    const { error } = await supabase
      .from('nodes')
      .update({ status, last_updated: new Date().toISOString() })
      .eq('id', nodeId);

    if (error) throw error;

    return { success: true };
  } catch (err: any) {
    console.error('❌ Error updating node status:', err);
    return {
      success: false,
      error: err.message,
    };
  }
}

/**
 * Report a new incident
 */
export async function reportIncident(
  incident: Omit<Incident, 'id' | 'timestamp' | 'created_at'>
): Promise<{ success: boolean; incident?: Incident; error?: string }> {
  try {
    const { data, error } = await supabase
      .from('incidents_and_events')
      .insert([
        {
          ...incident,
          timestamp: new Date().toISOString(),
          created_at: new Date().toISOString(),
        },
      ])
      .select()
      .single();

    if (error) throw error;

    return {
      success: true,
      incident: data as Incident,
    };
  } catch (err: any) {
    console.error('❌ Error reporting incident:', err);
    return {
      success: false,
      error: err.message,
    };
  }
}

// ============================================================================
// Batch Operations
// ============================================================================

/**
 * Load initial data payload for the application
 */
export async function loadInitialData(): Promise<{
  nodes: Node[];
  flows: Flow[];
  incidents: Incident[];
  macro_indicators: MacroIndicator[];
  error?: string;
}> {
  try {
    const [nodesRes, flowsRes, incidentsRes, indicatorsRes] = await Promise.all([
      fetchAllNodes(),
      fetchAllFlows(),
      fetchIncidents({ limit: 100 }),
      fetchMacroIndicators(),
    ]);

    return {
      nodes: nodesRes.nodes || [],
      flows: flowsRes.flows || [],
      incidents: incidentsRes.incidents || [],
      macro_indicators: indicatorsRes.indicators || [],
      error: nodesRes.error || flowsRes.error,
    };
  } catch (err: any) {
    console.error('❌ Error loading initial data:', err);
    return {
      nodes: [],
      flows: [],
      incidents: [],
      macro_indicators: [],
      error: err.message,
    };
  }
}

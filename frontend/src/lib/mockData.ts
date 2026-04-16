/**
 * Generate mock data for testing the spatial visualization
 * Nigeria-focused petrochemical supply chain data
 */

import { Node, Flow } from '@/types';

const MAJOR_NIGERIAN_CITIES = [
  { name: 'Lagos', lat: 6.5244, lng: 3.3792 },
  { name: 'Port Harcourt', lat: 4.7711, lng: 7.0314 },
  { name: 'Warri', lat: 5.5244, lng: 5.7394 },
  { name: 'Benin City', lat: 6.3350, lng: 5.6037 },
  { name: 'Abuja', lat: 9.0765, lng: 7.3986 },
  { name: 'Kaduna', lat: 10.5269, lng: 7.4337 },
  { name: 'Kano', lat: 12.0022, lng: 8.6753 },
  { name: 'Ibadan', lat: 7.3775, lng: 3.9470 },
  { name: 'Calabar', lat: 4.9500, lng: 8.3833 },
  { name: 'Akure', lat: 7.2543, lng: 5.1948 },
];

const NODE_TYPES = ['refinery', 'upstream_field', 'terminal', 'jetty', 'pipeline', 'retail_station', 'distribution_center'];
const STATUSES = ['operational', 'maintenance', 'degraded'];
const PRODUCT_TYPES = ['crude_oil', 'gasoline', 'diesel', 'kerosene', 'lpg'];
const TRANSPORT_MODES = ['pipeline', 'truck', 'vessel'];

function randomChoice<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function randomBetween(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

function generateNodeId(type: string, idx: number): string {
  return `node-${type}-${idx}`;
}

export function generateMockNodes(count: number = 45): Node[] {
  const nodes: Node[] = [];
  const nodeTypeCount: Record<string, number> = {};

  for (let i = 0; i < count; i++) {
    const type = randomChoice(NODE_TYPES);
    nodeTypeCount[type] = (nodeTypeCount[type] || 0) + 1;
    
    const city = randomChoice(MAJOR_NIGERIAN_CITIES);
    const latOffset = randomBetween(-0.5, 0.5);
    const lngOffset = randomBetween(-0.5, 0.5);

    nodes.push({
      id: generateNodeId(type, nodeTypeCount[type]),
      name: `${city.name} ${type.replace('_', ' ').toUpperCase()} ${nodeTypeCount[type]}`,
      node_type: type,
      latitude: city.lat + latOffset,
      longitude: city.lng + lngOffset,
      status: randomChoice(STATUSES),
      state: 'Nigeria',
      lga: city.name,
      capacity_bpd: Math.floor(randomBetween(10000, 500000)),
      connected_flows: Math.floor(randomBetween(1, 15)),
      last_updated: new Date().toISOString(),
      metadata: {
        operational_since: 2015 + Math.floor(randomBetween(0, 8)),
        efficiency: Math.round(randomBetween(70, 98)),
        incidents_30d: Math.floor(randomBetween(0, 3)),
      },
    });
  }

  return nodes;
}

export function generateMockFlows(nodes: Node[], count: number = 35): Flow[] {
  const flows: Flow[] = [];
  if (nodes.length < 2) return flows;

  for (let i = 0; i < count; i++) {
    const source = nodes[Math.floor(Math.random() * nodes.length)];
    
    // Find nearby nodes (within 2 degrees) for better connectivity
    const nearby = nodes.filter(n => {
      if (n.id === source.id) return false;
      const distance = Math.sqrt(
        Math.pow(n.latitude - source.latitude, 2) + 
        Math.pow(n.longitude - source.longitude, 2)
      );
      return distance < 2;  // Reduced from 5 to 2 degrees for tighter clusters
    });

    // Also consider some long-distance connections for inter-regional flows
    if (nearby.length === 0) {
      // Create cross-regional connections
      const randomDestination = nodes[Math.floor(Math.random() * nodes.length)];
      if (randomDestination.id !== source.id) {
        nearby.push(randomDestination);
      }
    }

    if (nearby.length === 0) continue;

    const destination = nearby[Math.floor(Math.random() * nearby.length)];

    flows.push({
      id: `flow-${i}`,
      source_node_id: source.id,
      destination_node_id: destination.id,
      transport_mode: randomChoice(TRANSPORT_MODES),
      flow_type: 'pipeline',
      product_type: randomChoice(PRODUCT_TYPES),
      volume_bpd: Math.floor(randomBetween(5000, 100000)),
      status: randomChoice(STATUSES),
      last_updated: new Date().toISOString(),
      is_active: true,
      avg_volume_value: Math.floor(randomBetween(5000, 100000)),
      metadata: {
        distance_km: Math.round(
          randomBetween(10, 300)
        ),
        pressure_bar: Math.round(randomBetween(20, 120)),
        utilization: Math.round(randomBetween(40, 95)),
      },
    });
  }

  return flows;
}

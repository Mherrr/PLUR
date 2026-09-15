import { ScatterplotLayer, PathLayer, PolygonLayer, TextLayer } from '@deck.gl/layers'
import { HeatmapLayer } from '@deck.gl/aggregation-layers'

// ── venue colour palette ────────────────────────────────────────────────────

const STAGE_COLORS = {
  hard:        [220,  50,  50, 255],
  harder:      [255, 120,  30, 255],
  green:       [ 50, 200,  80, 255],
  purple:      [160,  80, 220, 255],
  pink:        [240,  80, 160, 255],
  beatbox:     [ 80, 200, 240, 255],
  locals_only: [255, 210,   0, 255],
}
const DEFAULT_STAGE_COLOR = [180, 180, 180, 255]

function stageColor(id) {
  return STAGE_COLORS[id] ?? DEFAULT_STAGE_COLOR
}

const STAGE_NAME_TO_ID = {
  hard: 'hard', harder: 'harder', green: 'green',
  purple: 'purple', pink: 'pink',
}

function bohStageId(name) {
  const lower = (name || '').toLowerCase()
  for (const key of Object.keys(STAGE_NAME_TO_ID)) {
    if (lower.includes(key)) return STAGE_NAME_TO_ID[key]
  }
  return null
}

function bohColor(name) {
  const id = bohStageId(name)
  if (!id) return [35, 35, 42, 210]
  const c = stageColor(id)
  return [Math.round(c[0] * 0.45), Math.round(c[1] * 0.45), Math.round(c[2] * 0.45), 220]
}

function bohLineColor(name) {
  const id = bohStageId(name)
  if (!id) return [70, 70, 80, 120]
  const c = stageColor(id)
  return [c[0], c[1], c[2], 180]
}

function obstacleColor(subtype) {
  switch (subtype) {
    case 'structure':  return [55, 55, 65, 220]
    case 'terrain':    return [20, 35, 70, 210]
    case 'barrier':    return [90, 30, 30, 200]
    case 'bar':        return [60, 40, 20, 180]
    case 'vendor':     return [40, 40, 25, 180]
    default:           return [50, 50, 55, 180]
  }
}

function polyCentroid(coords) {
  const pts = coords[0] || coords
  let sx = 0, sy = 0, n = 0
  for (const [x, y] of pts) { sx += x; sy += y; n++ }
  return n > 0 ? [sx / n, sy / n] : [0, 0]
}

function stageLabelFromBoh(name) {
  return (name || '').replace(/\s*Back-of-House\s*/i, '').trim()
}

// ── venue layers ─────────────────────────────────────────────────────────────

// amenities = array of { id, name, facility_type, lon, lat } managed externally.
// When provided, the static geojson facility dots are skipped in favour of
// the dynamic buildAmenityLayers() output rendered separately in DeckMap.
export function buildVenueLayers(geojson, amenities = undefined) {
  if (!geojson?.features) return []

  const features   = geojson.features
  const walkable   = features.filter(f => f.properties.type === 'walkable')
  const boh        = features.filter(f => f.properties.type === 'obstacle' && f.properties.subtype === 'stage_boh')
  const obstacles  = features.filter(f => (f.properties.type === 'obstacle' && f.properties.subtype !== 'stage_boh' && f.properties.subtype !== 'bar') || f.properties.type === 'gate_area')
  const vip        = features.filter(f => f.properties.type === 'vip_area')
  const gates      = features.filter(f => f.properties.type === 'gate')
  const facilities = features.filter(f => f.properties.type === 'facility')

  return [
    new PolygonLayer({
      id: 'venue-walkable',
      data: walkable,
      getPolygon: f => f.geometry.coordinates[0],
      getFillColor: [18, 20, 26, 240],
      getLineColor: [60, 65, 80, 200],
      lineWidthMinPixels: 1.5,
      filled: true,
      stroked: true,
    }),

    new PolygonLayer({
      id: 'venue-vip',
      data: vip,
      getPolygon: f => f.geometry.coordinates[0],
      getFillColor: [70, 45, 110, 70],
      getLineColor: [140, 90, 200, 180],
      lineWidthMinPixels: 1,
      filled: true,
      stroked: true,
      pickable: true,
    }),

    new PolygonLayer({
      id: 'venue-obstacles',
      data: obstacles,
      getPolygon: f => f.geometry.coordinates[0],
      getFillColor: f => obstacleColor(f.properties.subtype),
      getLineColor: [70, 70, 80, 120],
      lineWidthMinPixels: 0.5,
      filled: true,
      stroked: true,
      pickable: true,
    }),

    new PolygonLayer({
      id: 'venue-boh',
      data: boh,
      getPolygon: f => f.geometry.coordinates[0],
      getFillColor: f => bohColor(f.properties.name),
      getLineColor: f => bohLineColor(f.properties.name),
      lineWidthMinPixels: 1.5,
      filled: true,
      stroked: true,
      pickable: true,
    }),

    new TextLayer({
      id: 'venue-stage-labels',
      data: boh,
      getPosition: f => [...polyCentroid(f.geometry.coordinates), 1],
      getText: f => stageLabelFromBoh(f.properties.name),
      getSize: 12,
      getColor: f => {
        const c = stageColor(bohStageId(f.properties.name))
        return [c[0], c[1], c[2], 255]
      },
      fontWeight: 700,
      fontFamily: 'Inter, system-ui, sans-serif',
      getTextAnchor: 'middle',
      getAlignmentBaseline: 'center',
      parameters: { depthTest: false },
    }),

    // Static facility dots only rendered when amenities are not managed externally
    ...(amenities === undefined ? [new ScatterplotLayer({
      id: 'venue-facilities',
      data: facilities,
      getPosition: f => f.geometry.coordinates,
      getRadius: 6,
      radiusUnits: 'meters',
      radiusMinPixels: 4,
      getFillColor: f => f.properties.facility_type === 'water'
        ? [30, 140, 220, 200]
        : [160, 160, 180, 200],
      getLineColor: [255, 255, 255, 120],
      stroked: true,
      lineWidthMinPixels: 1,
      pickable: true,
    })] : []),

    new ScatterplotLayer({
      id: 'venue-gates',
      data: gates,
      getPosition: f => [...f.geometry.coordinates, 2],
      getRadius: 10,
      radiusUnits: 'meters',
      radiusMinPixels: 6,
      getFillColor: [200, 230, 255, 220],
      getLineColor: [255, 255, 255, 255],
      stroked: true,
      lineWidthMinPixels: 2,
      pickable: true,
      parameters: { depthTest: false },
    }),

    new TextLayer({
      id: 'venue-gate-labels',
      data: gates,
      getPosition: f => [...f.geometry.coordinates, 2],
      getText: f => f.properties.name,
      getSize: 10,
      getColor: [180, 210, 255, 255],
      getPixelOffset: [0, 20],
      fontFamily: 'Inter, system-ui, sans-serif',
      getTextAnchor: 'middle',
      getAlignmentBaseline: 'center',
      parameters: { depthTest: false },
    }),
  ]
}

// ── simulation / risk layers ─────────────────────────────────────────────────

function speedColor(speed) {
  const t = Math.min(speed / 1.5, 1)
  return [Math.round(255 * (1 - t)), Math.round(50 * t), Math.round(255 * t), 210]
}

export function buildSimLayers({ agents, barriers, hotspots, vis }) {
  const layers = []

  if (vis.heatmap && agents.length > 0) {
    layers.push(new HeatmapLayer({
      id: 'heatmap',
      data: agents,
      getPosition: d => [d[0], d[1]],
      radiusPixels: 35,
      intensity: 1.5,
      threshold: 0.05,
      colorRange: [
        [0,   200, 100, 150],
        [255, 235,   0, 190],
        [255, 140,   0, 220],
        [220,  40,  40, 255],
      ],
    }))
  }

  if (vis.agents && agents.length > 0) {
    layers.push(new ScatterplotLayer({
      id: 'agents',
      data: agents,
      getPosition: d => [d[0], d[1]],
      getRadius: 2,
      radiusUnits: 'meters',
      radiusMinPixels: 1.5,
      getFillColor: d => speedColor(Math.hypot(d[2] ?? 0, d[3] ?? 0)),
      pickable: false,
    }))
  }

  if (barriers.length > 0) {
    layers.push(new PolygonLayer({
      id: 'user-barriers',
      data: barriers,
      getPolygon: d => d.polygon,
      getFillColor: [255, 210, 0, 60],
      getLineColor: [255, 210, 0, 230],
      lineWidthMinPixels: 2,
      filled: true,
      stroked: true,
      pickable: true,
    }))
  }

  if (vis.hotspots && hotspots.length > 0) {
    layers.push(new ScatterplotLayer({
      id: 'hotspots',
      data: hotspots,
      getPosition: d => [d.lon, d.lat],
      getRadius: 15,
      radiusUnits: 'meters',
      radiusMinPixels: 8,
      getFillColor: [0, 0, 0, 0],
      getLineColor: d => (d.level === 'orange' ? [255, 150, 0, 235] : [255, 40, 40, 255]),
      stroked: true,
      lineWidthMinPixels: 2.5,
      pickable: true,
    }))
  }

  return layers
}

// ── amenity layers ────────────────────────────────────────────────────────────

const AMENITY_FILL = {
  restroom: [160, 160, 180, 220],
  water:    [30,  140, 220, 220],
  bar:      [180, 120,  40, 220],
}

export function buildAmenityLayers(amenities, selectedAmenityId = null) {
  if (!amenities?.length) return []

  // Tag each item so DeckMap pick handlers can identify them by amenityId
  const data = amenities.map(a => ({ ...a, amenityId: a.id }))

  const layers = [
    new ScatterplotLayer({
      id: 'amenity-dots',
      data,
      getPosition: d => [d.lon, d.lat],
      getRadius: 6,
      radiusUnits: 'meters',
      radiusMinPixels: 5,
      getFillColor: d => AMENITY_FILL[d.facility_type] ?? [160, 160, 180, 220],
      getLineColor: [255, 255, 255, 130],
      stroked: true,
      lineWidthMinPixels: 1,
      pickable: true,
      parameters: { depthTest: false },
    }),
  ]

  if (selectedAmenityId) {
    const sel = data.find(a => a.id === selectedAmenityId)
    if (sel) {
      layers.push(new ScatterplotLayer({
        id: 'amenity-selected-ring',
        data: [sel],
        getPosition: d => [d.lon, d.lat],
        getRadius: 14,
        radiusUnits: 'meters',
        radiusMinPixels: 11,
        getFillColor: [255, 255, 255, 0],
        getLineColor: [255, 255, 255, 255],
        stroked: true,
        filled: false,
        lineWidthMinPixels: 2,
        parameters: { depthTest: false },
      }))
    }
  }

  return layers
}

import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import DeckMap from '../deck/DeckMap.jsx'
import { api } from '../api/client.js'

const STAGE_COLORS = {
  hard: '#ef4444', harder: '#f97316', green: '#22c55e',
  purple: '#a855f7', pink: '#ec4899', beatbox: '#22d3ee', locals_only: '#fbbf24',
}

const PANEL_W = 272

function generateSlots(startTime, endTime) {
  const parse = t => { const [h, m] = t.split(':').map(Number); return h * 60 + (m || 0) }
  let start = parse(startTime || '12:00')
  let end = parse(endTime || '00:00')
  if (end <= start) end += 24 * 60
  const slots = []
  for (let t = start; t < end; t += 60) {
    const h1 = Math.floor(t / 60) % 24
    const h2 = Math.floor((t + 60) / 60) % 24
    slots.push({
      label: `${h1 === 0 ? 12 : h1 > 12 ? h1 - 12 : h1}:00 ${h1 >= 12 ? 'PM' : 'AM'} – ${h2 === 0 ? 12 : h2 > 12 ? h2 - 12 : h2}:00 ${h2 >= 12 ? 'PM' : 'AM'}`,
      start: `${String(h1).padStart(2, '0')}:00`,
      end: `${String(h2).padStart(2, '0')}:00`,
    })
  }
  return slots
}

function setlistToSchedule(setlist, stages, slots) {
  const schedule = {}
  for (const stage of stages) {
    schedule[stage.id] = {}
    for (let i = 0; i < slots.length; i++) schedule[stage.id][i] = { artist: null, locked: false, manual: true, optimized: false }
  }
  for (const item of setlist ?? []) {
    const idx = slots.findIndex(s => s.start === item.start)
    if (idx !== -1 && schedule[item.stage]) {
      schedule[item.stage][idx] = { artist: item.artist, locked: item.locked ?? false, manual: item.manual ?? true, optimized: item.optimized ?? false }
    }
  }
  return schedule
}

function scheduleToSetlist(schedule, stages, slots) {
  const result = []
  for (const stage of stages) {
    for (let i = 0; i < slots.length; i++) {
      const cell = schedule[stage.id]?.[i]
      if (cell?.artist) {
        result.push({ artist: cell.artist, stage: stage.id, start: slots[i].start, end: slots[i].end, locked: cell.locked, manual: cell.manual ?? true, optimized: cell.optimized ?? false })
      }
    }
  }
  return result
}

export default function ProjectDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [project, setProject] = useState(null)
  const [loading, setLoading] = useState(true)
  const [setlist, setSetlist] = useState([])
  const [vis, setVis] = useState({
    heatmap: true, agents: false, hotspots: true,
  })
  const [simParams, setSimParams] = useState({
    capacity: 80000, tickets_sold: 60000,
    density_orange: 4, density_red: 6,
  })
  const [simRunning, setSimRunning] = useState(false)
  const [simFrames, setSimFrames] = useState([])
  const [simHotspots, setSimHotspots] = useState([])
  const [frameIdx, setFrameIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const playRef = useRef(null)
  const [showSetTimes, setShowSetTimes] = useState(false)
  const [addMode, setAddMode] = useState(null)
  const [userBarriers, setUserBarriers] = useState([])
  const [selectedBarrierId, setSelectedBarrierId] = useState(null)
  const [userAmenities, setUserAmenities] = useState([])
  const [selectedAmenityId, setSelectedAmenityId] = useState(null)
  const [showExitModal, setShowExitModal] = useState(false)
  const [optimizing, setOptimizing] = useState(false)
  const [optimizeResult, setOptimizeResult] = useState(null)
  const [briefingLoading, setBriefingLoading] = useState(false)
  const [briefingText, setBriefingText] = useState(null)

  useEffect(() => {
    api.getProject(id)
      .then(p => {
        setProject(p)
        setSetlist(p.setlist ?? [])
        if (p.meta?.barriers?.length > 0) setUserBarriers(p.meta.barriers)
        setSimParams(prev => ({
          ...prev,
          capacity: p.meta?.max_occupancy ?? 80000,
          tickets_sold: p.meta?.expected_attendance ?? Math.round((p.meta?.max_occupancy ?? 80000) * 0.8),
        }))
        return api.getSavedSim(id)
      })
      .then(sim => {
        if (sim?.frames?.length > 0) {
          setSimFrames(sim.frames)
          setSimHotspots(sim.hotspots || [])
          setFrameIdx(0)
        }
      })
      .catch(() => navigate('/'))
      .finally(() => setLoading(false))
  }, [id])

  // Initialize amenity positions — prefer saved positions from meta, fall back to GeoJSON
  useEffect(() => {
    if (!project?.geojson?.features) return
    if (project.meta?.amenities?.length > 0) {
      setUserAmenities(project.meta.amenities)
      return
    }
    setUserAmenities(
      project.geojson.features
        .filter(f => f.properties?.type === 'facility')
        .map(f => ({
          id: f.properties.facility_id ?? f.properties.id ?? f.properties.name?.toLowerCase().replace(/\s+/g, '_') ?? `amenity_${Math.random()}`,
          name: f.properties.name ?? f.properties.id ?? 'Facility',
          facility_type: f.properties.facility_type ?? 'restroom',
          lon: f.geometry.coordinates[0],
          lat: f.geometry.coordinates[1],
        }))
    )
  }, [project?.geojson])

  const stages = useMemo(() => {
    if (!project?.geojson) return []
    return (project.geojson.features ?? [])
      .filter(f => f.properties?.type === 'stage')
      .map(f => ({
        id: f.properties.stage_id ?? f.properties.id ?? f.properties.name?.toLowerCase().replace(/\s+/g, '_') ?? 'unknown',
        name: f.properties.name ?? f.properties.stage_id ?? 'Stage',
        color: STAGE_COLORS[f.properties.stage_id] ?? '#888',
      }))
  }, [project])

  const slots = useMemo(() => {
    if (!project?.meta) return []
    return generateSlots(project.meta.start_time ?? '12:00', project.meta.end_time ?? '00:00')
  }, [project])

  async function handleRunSim() {
    setSimRunning(true)
    setPlaying(false)
    setFrameIdx(0)
    setSimFrames([])
    setSimHotspots([])
    try {
      const barrierPolygons = userBarriers.map(b => barrierToPolygon(b))
      const result = await api.simulateFestival({
        projectId: id,
        setlist: setlist.map(s => ({
          artist: s.artist,
          stage: s.stage,
          start: s.start,
          end: s.end,
        })),
        sliders: {
          max_capacity: simParams.capacity,
          tickets_sold: simParams.tickets_sold,
          n_agents: Math.min(simParams.tickets_sold, 8000),
        },
        barriers: barrierPolygons,
        densityOrange: simParams.density_orange,
        densityRed: simParams.density_red,
      })
      setSimFrames(result.frames || [])
      setSimHotspots(result.hotspots || [])
      setFrameIdx(0)
    } catch (err) {
      alert('Simulation failed: ' + err.message)
    } finally {
      setSimRunning(false)
    }
  }

  // Playback loop — vary interval instead of step size for smooth slow-mo
  useEffect(() => {
    if (playing && simFrames.length > 0) {
      const interval = Math.max(10, Math.round(80 / speed))
      playRef.current = setInterval(() => {
        setFrameIdx(prev => {
          if (prev >= simFrames.length - 1) {
            setPlaying(false)
            return prev
          }
          return prev + 1
        })
      }, interval)
    }
    return () => clearInterval(playRef.current)
  }, [playing, simFrames.length, speed])

  const currentFrame = simFrames[Math.min(frameIdx, simFrames.length - 1)] || null
  const currentAgents = currentFrame?.agents || []
  const timelinePct = simFrames.length > 1 ? frameIdx / (simFrames.length - 1) : 0

  async function handleSaveQuit() {
    try {
      await api.updateProject(id, {
        setlist,
        meta: { ...project.meta, barriers: userBarriers, amenities: userAmenities },
      })
    } catch {}
    navigate('/')
  }

  async function handleSetTimesSave(newSetlist, newArtists) {
    setSetlist(newSetlist)
    setProject(prev => ({ ...prev, artists: newArtists }))
    try { await api.updateProject(id, { setlist: newSetlist, artists: newArtists }) } catch {}
    setShowSetTimes(false)
  }

  function handleMapClick(coord) {
    if (addMode !== 'barrier') return
    const barrier = {
      barrierId: `barrier_${Date.now()}`,
      cx: coord[0], cy: coord[1],
      hw: 0.0003, hh: 0.0002,
      angle: 0,
    }
    setUserBarriers(prev => [...prev, barrier])
    setAddMode(null)
  }

  function handleRemoveBarrier() {
    if (!selectedBarrierId) return
    setUserBarriers(prev => prev.filter(b => b.barrierId !== selectedBarrierId))
    setSelectedBarrierId(null)
  }

  async function handleOptimize() {
    if (setlist.length < 2) return alert('Need at least 2 artists in the schedule to optimize.')
    setOptimizing(true)
    setOptimizeResult(null)
    try {
      const headliners = setlist.filter(s => s.locked).map(s => s.artist)
      const result = await api.optimizeSchedule(
        id,
        setlist.map(s => ({ artist: s.artist, stage: s.stage, start: s.start, end: s.end })),
        headliners,
        { max_capacity: simParams.capacity, tickets_sold: simParams.tickets_sold },
      )
      setOptimizeResult(result)
    } catch (err) {
      alert('Optimization failed: ' + err.message)
    } finally {
      setOptimizing(false)
    }
  }

  function handleAcceptOptimization() {
    if (!optimizeResult?.proposed_schedule) return
    const changedArtists = new Set((optimizeResult.changes ?? []).map(c => c.artist))
    const newSetlist = optimizeResult.proposed_schedule.map(s => ({
      ...s,
      locked: setlist.find(o => o.artist === s.artist)?.locked ?? false,
      manual: setlist.find(o => o.artist === s.artist)?.manual ?? true,
      optimized: changedArtists.has(s.artist),
    }))
    setSetlist(newSetlist)
    api.updateProject(id, { setlist: newSetlist }).catch(() => {})
    setOptimizeResult(null)
  }

  async function handleSafetyBriefing() {
    setBriefingLoading(true)
    setBriefingText(null)
    try {
      const peakDensity = simHotspots.reduce((max, h) => Math.max(max, h.peak_density ?? 0), 0)
      const result = await api.getSafetyBriefing(
        id,
        setlist.map(s => ({ artist: s.artist, stage: s.stage, start: s.start, end: s.end })),
        { max_capacity: simParams.capacity, tickets_sold: simParams.tickets_sold },
        peakDensity,
        simHotspots,
        userAmenities,
      )
      setBriefingText(result.briefing)
    } catch (err) {
      alert('Failed to generate briefing: ' + err.message)
    } finally {
      setBriefingLoading(false)
    }
  }

  function handleBarrierUpdate(barrierId, updates) {
    setUserBarriers(prev => prev.map(b =>
      b.barrierId === barrierId ? { ...b, ...updates } : b
    ))
  }

  function handleAmenityUpdate(amenityId, updates) {
    setUserAmenities(prev => prev.map(a => a.id === amenityId ? { ...a, ...updates } : a))
  }

  function handleBarrierClick(id) {
    setSelectedBarrierId(id)
    if (id) setSelectedAmenityId(null)
  }

  function handleAmenityClick(id) {
    setSelectedAmenityId(id)
    if (id) setSelectedBarrierId(null)
  }

  const barriersWithPolygons = useMemo(() =>
    userBarriers.map(b => ({ ...b, polygon: barrierToPolygon(b) })),
    [userBarriers],
  )

  if (loading) {
    return (
      <div style={{
        width: '100vw', height: '100vh', background: '#09090b',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: '#52525b', fontFamily: 'Inter, system-ui, sans-serif', fontSize: 14,
      }}>
        Loading project…
      </div>
    )
  }

  if (!project) return null

  return (
    <div style={{
      width: '100vw', height: '100vh', background: '#09090b',
      position: 'relative', overflow: 'hidden',
      fontFamily: 'Inter, system-ui, sans-serif', color: '#f4f4f5',
    }}>
      <DeckMap
        venueGeoJSON={project.geojson}
        agents={currentAgents} barriers={barriersWithPolygons} hotspots={simHotspots}
        vis={vis}
        addMode={addMode}
        onMapClick={handleMapClick}
        selectedBarrierId={selectedBarrierId}
        onBarrierClick={handleBarrierClick}
        onBarrierUpdate={handleBarrierUpdate}
        amenities={userAmenities}
        selectedAmenityId={selectedAmenityId}
        onAmenityClick={handleAmenityClick}
        onAmenityUpdate={handleAmenityUpdate}
      />

      {/* Top bar */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, zIndex: 10,
        background: 'rgba(9,9,11,0.88)', backdropFilter: 'blur(14px)',
        borderBottom: '1px solid #27272a',
        display: 'flex', alignItems: 'center', padding: '0 18px', height: 52, gap: 14,
      }}>
        <button
          onClick={() => setShowExitModal(true)}
          style={{
            background: 'none', border: '1px solid #3f3f46', borderRadius: 6,
            padding: '5px 12px', color: '#a1a1aa', cursor: 'pointer',
            fontSize: 13, fontFamily: 'inherit',
          }}
        >
          ← Home
        </button>
        <div style={{ width: 1, height: 20, background: '#27272a' }} />
        <span style={{ fontWeight: 700, color: '#f4f4f5', fontSize: 15 }}>{project.name}</span>
        <div style={{ marginLeft: 'auto' }}>
          <button
            onClick={handleSaveQuit}
            style={{
              background: '#18181b', border: '1px solid #3f3f46', borderRadius: 6,
              padding: '6px 16px', color: '#a1a1aa', cursor: 'pointer',
              fontSize: 13, fontFamily: 'inherit', fontWeight: 500,
            }}
          >
            Save & Quit
          </button>
        </div>
      </div>

      {/* Exit modal */}
      {showExitModal && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 100,
          background: 'rgba(0,0,0,0.78)', backdropFilter: 'blur(4px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{
            background: '#18181b', border: '1px solid #27272a', borderRadius: 14,
            padding: 32, maxWidth: 420, width: '90%',
          }}>
            <div style={{ fontWeight: 700, fontSize: 18, marginBottom: 10, color: '#f4f4f5' }}>
              Return home?
            </div>
            <p style={{ color: '#a1a1aa', fontSize: 14, margin: '0 0 24px' }}>
              Would you like to save your changes before leaving?
            </p>
            <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
              <button
                onClick={() => setShowExitModal(false)}
                style={{
                  background: 'transparent', color: '#a1a1aa', border: '1px solid #3f3f46',
                  borderRadius: 8, padding: '8px 18px', fontWeight: 500, fontSize: 13,
                  cursor: 'pointer', fontFamily: 'inherit',
                }}
              >
                Cancel
              </button>
              <button
                onClick={() => { setShowExitModal(false); navigate('/') }}
                style={{
                  background: 'transparent', color: '#f87171', border: '1px solid #7f1d1d',
                  borderRadius: 8, padding: '8px 18px', fontWeight: 500, fontSize: 13,
                  cursor: 'pointer', fontFamily: 'inherit',
                }}
              >
                Don't Save
              </button>
              <button
                onClick={() => { setShowExitModal(false); handleSaveQuit() }}
                style={{
                  background: '#3b82f6', color: '#fff', border: 'none',
                  borderRadius: 8, padding: '8px 18px', fontWeight: 600, fontSize: 13,
                  cursor: 'pointer', fontFamily: 'inherit',
                }}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Left control panel */}
      <ControlPanel
        simParams={simParams}
        onParamsChange={setSimParams}
        vis={vis}
        onToggleVis={key => setVis(v => ({ ...v, [key]: !v[key] }))}
        simRunning={simRunning}
        hasResult={simFrames.length > 0}
        onRunSim={handleRunSim}
        onSetTimes={() => setShowSetTimes(true)}
        addMode={addMode}
        onAddBarrier={() => setAddMode(m => m === 'barrier' ? null : 'barrier')}
        selectedBarrierId={selectedBarrierId}
        onRemoveBarrier={handleRemoveBarrier}
        onDeselectBarrier={() => handleBarrierClick(null)}
        barrierCount={userBarriers.length}
        userAmenities={userAmenities}
        selectedAmenityId={selectedAmenityId}
        onDeselectAmenity={() => setSelectedAmenityId(null)}
        optimizing={optimizing}
        onOptimize={handleOptimize}
        hasSetlist={setlist.length >= 2}
        briefingLoading={briefingLoading}
        onSafetyBriefing={handleSafetyBriefing}
        hasSimResult={simFrames.length > 0}
      />

      {/* Optimize results modal */}
      {optimizeResult && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 100,
          background: 'rgba(0,0,0,0.78)', backdropFilter: 'blur(4px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{
            background: '#18181b', border: '1px solid #27272a', borderRadius: 14,
            padding: 28, maxWidth: 520, width: '90%', maxHeight: '80vh', overflowY: 'auto',
          }}>
            <div style={{ fontWeight: 700, fontSize: 18, marginBottom: 6, color: '#f4f4f5' }}>
              Optimization Results
            </div>
            <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
              <div style={{
                background: '#09090b', borderRadius: 8, padding: '10px 14px', flex: 1,
                border: '1px solid #27272a',
              }}>
                <div style={{ fontSize: 10, color: '#71717a', marginBottom: 2 }}>RISK BEFORE</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: '#f87171' }}>
                  {optimizeResult.risk_before.toFixed(2)}
                </div>
              </div>
              <div style={{
                background: '#09090b', borderRadius: 8, padding: '10px 14px', flex: 1,
                border: '1px solid #27272a',
              }}>
                <div style={{ fontSize: 10, color: '#71717a', marginBottom: 2 }}>RISK AFTER</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: '#4ade80' }}>
                  {optimizeResult.risk_after.toFixed(2)}
                </div>
              </div>
              <div style={{
                background: '#09090b', borderRadius: 8, padding: '10px 14px', flex: 1,
                border: '1px solid #27272a',
              }}>
                <div style={{ fontSize: 10, color: '#71717a', marginBottom: 2 }}>REDUCTION</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: '#60a5fa' }}>
                  {optimizeResult.risk_before > 0
                    ? Math.round((1 - optimizeResult.risk_after / optimizeResult.risk_before) * 100) + '%'
                    : '—'}
                </div>
              </div>
            </div>

            {optimizeResult.changes.length > 0 ? (
              <>
                <div style={{ fontSize: 12, color: '#a1a1aa', marginBottom: 8, fontWeight: 600 }}>
                  Proposed Changes ({optimizeResult.changes.length})
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 16 }}>
                  {optimizeResult.changes.map((c, i) => (
                    <div key={i} style={{
                      background: '#09090b', border: '1px solid #27272a', borderRadius: 6,
                      padding: '8px 12px', fontSize: 12,
                    }}>
                      <span style={{ color: '#f4f4f5', fontWeight: 600 }}>{c.artist}</span>
                      <span style={{ color: '#71717a' }}> — </span>
                      <span style={{ color: '#f87171' }}>{c.from_stage} {c.from_time}</span>
                      <span style={{ color: '#71717a' }}> → </span>
                      <span style={{ color: '#4ade80' }}>{c.to_stage} {c.to_time}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p style={{ color: '#71717a', fontSize: 13, marginBottom: 16 }}>
                Schedule is already optimal — no changes needed.
              </p>
            )}

            {optimizeResult.rationale && (
              <div style={{
                background: '#09090b', border: '1px solid #27272a', borderRadius: 8,
                padding: '10px 14px', marginBottom: 16, fontSize: 12, color: '#a1a1aa',
                lineHeight: 1.5,
              }}>
                {optimizeResult.rationale}
              </div>
            )}

            <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
              <button
                onClick={() => setOptimizeResult(null)}
                style={{
                  background: 'transparent', color: '#a1a1aa', border: '1px solid #3f3f46',
                  borderRadius: 8, padding: '8px 18px', fontWeight: 500, fontSize: 13,
                  cursor: 'pointer', fontFamily: 'inherit',
                }}
              >
                Discard
              </button>
              {optimizeResult.changes.length > 0 && (
                <button
                  onClick={handleAcceptOptimization}
                  style={{
                    background: '#3b82f6', color: '#fff', border: 'none',
                    borderRadius: 8, padding: '8px 18px', fontWeight: 600, fontSize: 13,
                    cursor: 'pointer', fontFamily: 'inherit',
                  }}
                >
                  Accept Changes
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Safety briefing modal */}
      {briefingText && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 100,
          background: 'rgba(0,0,0,0.78)', backdropFilter: 'blur(4px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{
            background: '#18181b', border: '1px solid #27272a', borderRadius: 14,
            padding: 28, maxWidth: 560, width: '90%', maxHeight: '80vh', overflowY: 'auto',
          }}>
            <div style={{ fontWeight: 700, fontSize: 18, marginBottom: 16, color: '#f4f4f5' }}>
              Safety Briefing
            </div>
            <div style={{
              whiteSpace: 'pre-wrap', fontSize: 13, color: '#d4d4d8',
              lineHeight: 1.6, fontFamily: 'inherit',
            }}>
              {briefingText}
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 20 }}>
              <button
                onClick={() => setBriefingText(null)}
                style={{
                  background: 'transparent', color: '#a1a1aa', border: '1px solid #3f3f46',
                  borderRadius: 8, padding: '8px 18px', fontWeight: 500, fontSize: 13,
                  cursor: 'pointer', fontFamily: 'inherit',
                }}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Timeline scrubber — always visible, dimmed until sim runs */}
      <TimelineBar
        meta={project.meta}
        playing={playing}
        onPlayPause={() => { if (simFrames.length > 0) setPlaying(p => !p) }}
        speed={speed}
        onSpeed={setSpeed}
        value={timelinePct}
        onChange={pct => {
          setFrameIdx(Math.round(pct * Math.max(1, simFrames.length - 1)))
          setPlaying(false)
        }}
        disabled={simFrames.length === 0}
        currentFrame={currentFrame}
      />

      {/* Set Times overlay */}
      {showSetTimes && (
        <SetTimesOverlay
          stages={stages}
          slots={slots}
          setlist={setlist}
          artists={project.artists ?? []}
          onSave={handleSetTimesSave}
          onClose={() => setShowSetTimes(false)}
        />
      )}
    </div>
  )
}

function barrierToPolygon(b) {
  const cos = Math.cos(b.angle)
  const sin = Math.sin(b.angle)
  const corners = [
    [-b.hw, -b.hh],
    [ b.hw, -b.hh],
    [ b.hw,  b.hh],
    [-b.hw,  b.hh],
  ]
  const pts = corners.map(([dx, dy]) => [
    b.cx + dx * cos - dy * sin,
    b.cy + dx * sin + dy * cos,
  ])
  return [...pts, pts[0]]
}

function sliderBg(value, min, max, color) {
  const pct = ((value - min) / (max - min)) * 100
  return `linear-gradient(to right, ${color} ${pct}%, #27272a ${pct}%)`
}

// ── ControlPanel ──────────────────────────────────────────────────────────────

function PanelSection({ label, children }) {
  return (
    <div style={{ padding: '0 16px 16px' }}>
      <div style={{
        fontSize: 10, fontWeight: 700, color: '#52525b',
        textTransform: 'uppercase', letterSpacing: '0.12em',
        marginBottom: 10, paddingTop: 14,
        borderTop: '1px solid #1c1c1e',
      }}>
        {label}
      </div>
      {children}
    </div>
  )
}

function ControlPanel({
  simParams, onParamsChange, vis, onToggleVis,
  simRunning, hasResult, onRunSim,
  onSetTimes, addMode, onAddBarrier,
  selectedBarrierId, onRemoveBarrier, onDeselectBarrier, barrierCount,
  userAmenities, selectedAmenityId, onDeselectAmenity,
  optimizing, onOptimize, hasSetlist,
  briefingLoading, onSafetyBriefing, hasSimResult,
}) {
  const ticketsMax = simParams.capacity || 80000
  const overCapacity = simParams.tickets_sold > simParams.capacity

  return (
    <div style={{
      position: 'absolute', top: 52, left: 0, bottom: 0, width: PANEL_W,
      background: 'rgba(9,9,11,0.93)', backdropFilter: 'blur(14px)',
      borderRight: '1px solid #27272a',
      zIndex: 10, overflowY: 'auto',
      display: 'flex', flexDirection: 'column',
    }}>
      {/* Run simulation */}
      <div style={{ padding: '16px 16px 0' }}>
        <button
          onClick={onRunSim}
          disabled={simRunning}
          style={{
            width: '100%', background: simRunning ? '#1e3a5f' : '#2563eb',
            color: '#fff', border: 'none', borderRadius: 8,
            padding: '11px 0', fontWeight: 700, fontSize: 14,
            cursor: simRunning ? 'not-allowed' : 'pointer', fontFamily: 'inherit',
            transition: 'background 0.15s',
          }}
        >
          {simRunning ? 'Running…' : hasResult ? 'Rerun Simulation' : 'Run Simulation'}
        </button>
      </div>

      <PanelSection label="Parameters">
        <label style={{ display: 'flex', flexDirection: 'column', gap: 5, marginBottom: 14 }}>
          <span style={{ fontSize: 12, color: '#a1a1aa', display: 'flex', justifyContent: 'space-between' }}>
            Capacity
            <span style={{ color: '#71717a', fontSize: 11 }}>hard cap</span>
          </span>
          <input
            type="number"
            value={simParams.capacity}
            onChange={e => onParamsChange(p => ({ ...p, capacity: Number(e.target.value) }))}
            style={{
              background: '#111113', border: '1px solid #3f3f46', borderRadius: 6,
              padding: '6px 10px', color: '#f4f4f5', fontSize: 13,
              fontFamily: 'inherit', outline: 'none', width: '100%', boxSizing: 'border-box',
            }}
          />
        </label>

        <label style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          <span style={{ fontSize: 12, display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: overCapacity ? '#ef4444' : '#a1a1aa' }}>Tickets Sold</span>
            <span style={{ color: overCapacity ? '#ef4444' : '#60a5fa', fontVariantNumeric: 'tabular-nums' }}>
              {Number(simParams.tickets_sold).toLocaleString()}
              {overCapacity && ' ⚠'}
            </span>
          </span>
          <input
            type="range"
            min={0} max={ticketsMax} step={100}
            value={Math.min(simParams.tickets_sold, ticketsMax)}
            onChange={e => onParamsChange(p => ({ ...p, tickets_sold: Number(e.target.value) }))}
            style={{ width: '100%', background: sliderBg(Math.min(simParams.tickets_sold, ticketsMax), 0, ticketsMax, overCapacity ? '#ef4444' : '#3b82f6') }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#3f3f46' }}>
            <span>0</span>
            <span>{Number(ticketsMax).toLocaleString()}</span>
          </div>
        </label>
      </PanelSection>

      <PanelSection label="Risk Thresholds">
        <label style={{ display: 'flex', flexDirection: 'column', gap: 5, marginBottom: 14 }}>
          <span style={{ fontSize: 12, color: '#f97316', display: 'flex', justifyContent: 'space-between' }}>
            Orange density
            <span style={{ color: '#a1a1aa' }}>{simParams.density_orange} p/m²</span>
          </span>
          <input
            type="range" min={1} max={8} step={0.5}
            value={simParams.density_orange}
            onChange={e => onParamsChange(p => ({ ...p, density_orange: Number(e.target.value) }))}
            style={{ width: '100%', background: sliderBg(simParams.density_orange, 1, 8, '#f97316') }}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          <span style={{ fontSize: 12, color: '#ef4444', display: 'flex', justifyContent: 'space-between' }}>
            Red density
            <span style={{ color: '#a1a1aa' }}>{simParams.density_red} p/m²</span>
          </span>
          <input
            type="range" min={1} max={8} step={0.5}
            value={simParams.density_red}
            onChange={e => onParamsChange(p => ({ ...p, density_red: Number(e.target.value) }))}
            style={{ width: '100%', background: sliderBg(simParams.density_red, 1, 8, '#ef4444') }}
          />
        </label>
      </PanelSection>

      <PanelSection label="Schedule">
        <button
          onClick={onSetTimes}
          style={{
            width: '100%', background: '#18181b', border: '1px solid #3f3f46',
            borderRadius: 7, padding: '9px 0', color: '#a1a1aa',
            cursor: 'pointer', fontFamily: 'inherit', fontSize: 13, fontWeight: 500,
          }}
        >
          Set Times →
        </button>
      </PanelSection>

      <PanelSection label={`Barriers (${barrierCount})`}>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={onAddBarrier}
            style={{
              flex: 1, background: addMode === 'barrier' ? '#1e3a5f' : '#18181b',
              border: `1px solid ${addMode === 'barrier' ? '#2563eb' : '#3f3f46'}`,
              borderRadius: 7, padding: '8px 0',
              color: addMode === 'barrier' ? '#60a5fa' : '#a1a1aa',
              cursor: 'pointer', fontFamily: 'inherit', fontSize: 12, fontWeight: 500,
            }}
          >
            {addMode === 'barrier' ? 'Click map to place…' : '+ Place Barrier'}
          </button>
        </div>
        {selectedBarrierId && (
          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
            <button
              onClick={onRemoveBarrier}
              style={{
                flex: 1, background: '#18181b', border: '1px solid #7f1d1d',
                borderRadius: 7, padding: '8px 0',
                color: '#f87171', cursor: 'pointer', fontFamily: 'inherit', fontSize: 12, fontWeight: 500,
              }}
            >
              Remove Selected
            </button>
            <button
              onClick={onDeselectBarrier}
              style={{
                background: '#18181b', border: '1px solid #3f3f46',
                borderRadius: 7, padding: '8px 12px',
                color: '#a1a1aa', cursor: 'pointer', fontFamily: 'inherit', fontSize: 12, fontWeight: 500,
              }}
            >
              Deselect
            </button>
          </div>
        )}
      </PanelSection>

      <PanelSection label={`Amenities (${userAmenities?.length ?? 0})`}>
        <p style={{ fontSize: 10, color: '#52525b', margin: '0 0 8px', lineHeight: 1.4 }}>
          Click to select, drag on the map to reposition.
        </p>
        {selectedAmenityId ? (() => {
          const am = userAmenities?.find(a => a.id === selectedAmenityId)
          const COLOR = { water: '#60a5fa', restroom: '#a1a1aa', bar: '#d97706' }
          const LABEL = { water: 'Water Station', restroom: 'Restrooms', bar: 'Bar' }
          return (
            <div>
              <div style={{ fontSize: 11, color: COLOR[am?.facility_type] ?? '#a1a1aa', marginBottom: 8 }}>
                {am?.name ?? selectedAmenityId}
                {am?.facility_type && (
                  <span style={{ color: '#52525b', marginLeft: 6 }}>
                    {LABEL[am.facility_type] ?? am.facility_type}
                  </span>
                )}
              </div>
              <button
                onClick={onDeselectAmenity}
                style={{
                  width: '100%', background: '#18181b', border: '1px solid #3f3f46',
                  borderRadius: 7, padding: '8px 0',
                  color: '#a1a1aa', cursor: 'pointer', fontFamily: 'inherit', fontSize: 12, fontWeight: 500,
                }}
              >
                Deselect
              </button>
            </div>
          )
        })() : null}
      </PanelSection>

      <PanelSection label="Optimize">
        <button
          onClick={onOptimize}
          disabled={optimizing || !hasSetlist}
          style={{
            width: '100%',
            background: optimizing ? '#1e3a5f' : '#18181b',
            border: `1px solid ${optimizing ? '#2563eb' : '#3f3f46'}`,
            borderRadius: 7, padding: '9px 12px',
            color: optimizing ? '#60a5fa' : hasSetlist ? '#a1a1aa' : '#52525b',
            cursor: optimizing || !hasSetlist ? 'not-allowed' : 'pointer',
            fontFamily: 'inherit', fontSize: 12, fontWeight: 500, textAlign: 'center',
          }}
        >
          {optimizing ? 'Optimizing…' : 'Optimize ▶'}
        </button>
        <p style={{ fontSize: 10, color: '#52525b', margin: '6px 0 0', lineHeight: 1.4 }}>
          Rearranges unlocked artists across stages/times to reduce crowd-crush risk. Lock artists in Set Times to keep them fixed.
        </p>
      </PanelSection>

      <PanelSection label="Safety Briefing">
        <button
          onClick={onSafetyBriefing}
          disabled={briefingLoading || !hasSimResult}
          style={{
            width: '100%',
            background: briefingLoading ? '#1e3a5f' : '#18181b',
            border: `1px solid ${briefingLoading ? '#2563eb' : '#3f3f46'}`,
            borderRadius: 7, padding: '9px 12px',
            color: briefingLoading ? '#60a5fa' : hasSimResult ? '#a1a1aa' : '#52525b',
            cursor: briefingLoading || !hasSimResult ? 'not-allowed' : 'pointer',
            fontFamily: 'inherit', fontSize: 12, fontWeight: 500, textAlign: 'center',
          }}
        >
          {briefingLoading ? 'Generating…' : 'Generate Briefing'}
        </button>
        {!hasSimResult && (
          <p style={{ fontSize: 10, color: '#52525b', margin: '6px 0 0', lineHeight: 1.4 }}>
            Run a simulation first to generate a safety briefing.
          </p>
        )}
      </PanelSection>

      <PanelSection label="Layers">
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {Object.entries(vis).map(([key, on]) => (
            <button
              key={key}
              onClick={() => onToggleVis(key)}
              style={{
                background: on ? '#1e3a5f' : '#18181b',
                color: on ? '#60a5fa' : '#52525b',
                border: `1px solid ${on ? '#2563eb' : '#27272a'}`,
                borderRadius: 5, padding: '3px 9px', fontSize: 11,
                cursor: 'pointer', fontFamily: 'inherit', fontWeight: 600,
              }}
            >
              {key === 'agents' ? 'Guests' : key.charAt(0).toUpperCase() + key.slice(1)}
            </button>
          ))}
        </div>
      </PanelSection>
    </div>
  )
}

// ── TimelineBar ───────────────────────────────────────────────────────────────

const SPEEDS = [0.25, 0.5, 1, 2, 4]

function TimelineBar({ meta, playing, onPlayPause, speed, onSpeed, value, onChange, disabled, currentFrame }) {
  function formatTime(t_min) {
    if (t_min == null) return '--:-- --'
    const h24 = Math.floor(((t_min % 1440) + 1440) % 1440 / 60)
    const m = Math.floor(t_min % 60)
    const ampm = h24 >= 12 ? 'PM' : 'AM'
    const h12 = h24 === 0 ? 12 : h24 > 12 ? h24 - 12 : h24
    return `${h12}:${String(m).padStart(2, '0')} ${ampm}`
  }
  const timeLabel = currentFrame ? formatTime(currentFrame.t_min) : '--:--'
  const agentInfo = currentFrame
    ? `${currentFrame.n_active} on site · ${currentFrame.n_exited} exited`
    : ''

  return (
    <div style={{
      position: 'absolute', bottom: 0, left: PANEL_W, right: 0, zIndex: 10,
      background: 'rgba(9,9,11,0.93)', backdropFilter: 'blur(14px)',
      borderTop: '1px solid #27272a',
      padding: '10px 18px',
      display: 'flex', alignItems: 'center', gap: 12,
      opacity: disabled ? 0.32 : 1,
      pointerEvents: disabled ? 'none' : 'auto',
      transition: 'opacity 0.2s',
    }}>
      <button
        onClick={onPlayPause}
        style={{
          background: 'none', border: '1px solid #3f3f46', borderRadius: 6,
          color: '#a1a1aa', cursor: 'pointer', padding: '4px 11px',
          fontSize: 14, fontFamily: 'inherit', lineHeight: 1, flexShrink: 0,
        }}
      >
        {playing ? '⏸' : '▶'}
      </button>

      <input
        type="range" min={0} max={1} step={0.001}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ flex: 1, background: sliderBg(value, 0, 1, '#3b82f6') }}
      />

      <span style={{ fontSize: 12, color: '#f4f4f5', fontVariantNumeric: 'tabular-nums', minWidth: 42, flexShrink: 0, fontWeight: 600 }}>
        {timeLabel}
      </span>
      {agentInfo && (
        <span style={{ fontSize: 10, color: '#71717a', minWidth: 120, flexShrink: 0 }}>
          {agentInfo}
        </span>
      )}

      <select
        value={speed}
        onChange={e => onSpeed(Number(e.target.value))}
        style={{
          background: '#18181b', border: '1px solid #3f3f46', borderRadius: 5,
          color: '#a1a1aa', fontSize: 11, padding: '3px 6px',
          fontFamily: 'inherit', cursor: 'pointer', flexShrink: 0,
        }}
      >
        {SPEEDS.map(s => <option key={s} value={s}>{s}x</option>)}
      </select>

      {disabled && (
        <span style={{ fontSize: 11, color: '#3f3f46', flexShrink: 0 }}>
          Run simulation to enable playback
        </span>
      )}
    </div>
  )
}

// ── SetTimesOverlay ───────────────────────────────────────────────────────────

function SetTimesOverlay({ stages, slots, setlist, artists: initialArtists, onSave, onClose }) {
  const [schedule, setSchedule] = useState(() => setlistToSchedule(setlist, stages, slots))
  const [artists, setArtists] = useState(initialArtists ?? [])
  // drag state: { artist, fromStage, fromSlot } — fromStage/fromSlot null = from sidebar
  const [drag, setDrag] = useState(null)
  const [dragOverCell, setDragOverCell] = useState(null)  // 'sidebar' | 'stage:idx'
  const [saving, setSaving] = useState(false)
  const [autoFilling, setAutoFilling] = useState(false)
  const [newArtistName, setNewArtistName] = useState('')

  const assignedSet = new Set(
    Object.values(schedule).flatMap(s => Object.values(s).map(c => c.artist).filter(Boolean))
  )
  const unassigned = artists.filter(a => a && !assignedSet.has(a))

  function startDragFrom(artist, fromStage = null, fromSlot = null) {
    setDrag({ artist, fromStage, fromSlot })
  }

  function dropOnCell(stageId, slotIdx) {
    if (!drag) return
    if (schedule[stageId]?.[slotIdx]?.locked) return
    const { artist, fromStage, fromSlot } = drag

    setSchedule(prev => {
      const next = JSON.parse(JSON.stringify(prev))
      const targetArtist = next[stageId][slotIdx].artist

      if (fromStage !== null && fromSlot !== null) {
        // swap: put displaced artist back in source — both become manual
        next[fromStage][fromSlot].artist = targetArtist ?? null
        next[fromStage][fromSlot].manual = true
        next[fromStage][fromSlot].optimized = false
      }
      if (fromStage === null) {
        for (const sid of Object.keys(next)) {
          for (const idx of Object.keys(next[sid])) {
            if (next[sid][idx].artist === artist) next[sid][idx].artist = null
          }
        }
      }
      next[stageId][slotIdx].artist = artist
      next[stageId][slotIdx].manual = true  // any user drag = manual
      next[stageId][slotIdx].optimized = false  // user override clears optimizer state
      return next
    })
    setDrag(null)
    setDragOverCell(null)
  }

  async function autoFill() {
    setAutoFilling(true)
    try {
      let drawScores = {}
      try {
        const res = await api.getDemandScores(artists)
        drawScores = res.draw ?? {}
      } catch {
        artists.forEach((a, i) => { drawScores[a] = i / artists.length })
      }

      const emptySlots = []
      for (const stage of stages) {
        for (let idx = 0; idx < slots.length; idx++) {
          if (!schedule[stage.id]?.[idx]?.artist) {
            emptySlots.push({ stageId: stage.id, slotIdx: idx, start: slots[idx].start })
          }
        }
      }
      emptySlots.sort((a, b) => a.start < b.start ? -1 : a.start > b.start ? 1 : a.stageId < b.stageId ? -1 : 1)

      const assigned = new Set(
        Object.values(schedule).flatMap(s => Object.values(s).map(c => c.artist).filter(Boolean))
      )
      const toPlace = artists
        .filter(a => a && !assigned.has(a))
        .sort((a, b) => (drawScores[a] ?? 0) - (drawScores[b] ?? 0))

      if (!toPlace.length || !emptySlots.length) return

      setSchedule(prev => {
        const next = JSON.parse(JSON.stringify(prev))
        const queue = [...emptySlots]
        for (const artist of toPlace) {
          if (!queue.length) break
          const { stageId, slotIdx } = queue.shift()
          next[stageId][slotIdx].artist = artist
          next[stageId][slotIdx].manual = false
          next[stageId][slotIdx].locked = false
          next[stageId][slotIdx].optimized = false
        }
        return next
      })
    } finally {
      setAutoFilling(false)
    }
  }

  function dropOnSidebar() {
    if (!drag || drag.fromStage === null) return
    const { fromStage, fromSlot } = drag
    if (schedule[fromStage]?.[fromSlot]?.locked) return
    setSchedule(prev => {
      const next = JSON.parse(JSON.stringify(prev))
      next[fromStage][fromSlot].artist = null
      return next
    })
    setDrag(null)
    setDragOverCell(null)
  }

  function toggleLock(stageId, slotIdx) {
    if (!schedule[stageId]?.[slotIdx]?.artist) return
    setSchedule(prev => ({
      ...prev,
      [stageId]: {
        ...prev[stageId],
        [slotIdx]: { ...prev[stageId][slotIdx], locked: !prev[stageId][slotIdx].locked },
      },
    }))
  }

  function addArtist() {
    const name = newArtistName.trim()
    if (!name || artists.includes(name)) return
    setArtists(prev => [...prev, name])
    setNewArtistName('')
  }

  function handleFileUpload(e) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const names = reader.result
        .split('\n')
        .map(l => l.trim())
        .filter(l => l.length > 0)
      const unique = names.filter(n => !artists.includes(n))
      if (unique.length > 0) setArtists(prev => [...prev, ...unique])
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  function removeArtist(name) {
    setArtists(prev => prev.filter(a => a !== name))
    setSchedule(prev => {
      const next = JSON.parse(JSON.stringify(prev))
      for (const sid of Object.keys(next)) {
        for (const idx of Object.keys(next[sid])) {
          if (next[sid][idx].artist === name) next[sid][idx].artist = null
        }
      }
      return next
    })
  }

  async function handleSave() {
    setSaving(true)
    await onSave(scheduleToSetlist(schedule, stages, slots), artists)
    setSaving(false)
  }

  const sidebarIsOver = dragOverCell === 'sidebar' && drag?.fromStage !== null

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 50,
      background: 'rgba(0,0,0,0.88)', backdropFilter: 'blur(6px)',
      display: 'flex', flexDirection: 'column',
      fontFamily: 'Inter, system-ui, sans-serif', color: '#f4f4f5',
    }}>
      {/* Header */}
      <div style={{
        height: 56, borderBottom: '1px solid #27272a',
        background: 'rgba(9,9,11,0.97)',
        display: 'flex', alignItems: 'center', padding: '0 24px', gap: 16, flexShrink: 0,
      }}>
        <span style={{ fontWeight: 700, fontSize: 16 }}>Set Times</span>
        <span style={{ fontSize: 12, color: '#52525b' }}>
          Drag artists between slots · double-click to lock · drop on sidebar to unassign
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 10, alignItems: 'center' }}>
          {unassigned.length > 0 && (
            <button
              onClick={autoFill}
              disabled={autoFilling}
              style={{
                background: 'transparent', border: '1px solid #16a34a', borderRadius: 6,
                padding: '6px 14px', color: autoFilling ? '#52525b' : '#4ade80',
                cursor: autoFilling ? 'not-allowed' : 'pointer',
                fontSize: 13, fontFamily: 'inherit', fontWeight: 500,
                opacity: autoFilling ? 0.6 : 1,
              }}
            >
              {autoFilling ? 'Filling…' : `Auto-fill ${unassigned.length} artist${unassigned.length !== 1 ? 's' : ''}`}
            </button>
          )}
          <button
            onClick={onClose}
            style={{
              background: 'none', border: '1px solid #3f3f46', borderRadius: 6,
              padding: '6px 16px', color: '#a1a1aa', cursor: 'pointer',
              fontSize: 13, fontFamily: 'inherit',
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleSave} disabled={saving}
            style={{
              background: '#2563eb', border: 'none', borderRadius: 6,
              padding: '6px 18px', color: '#fff',
              cursor: saving ? 'not-allowed' : 'pointer',
              fontSize: 13, fontFamily: 'inherit', fontWeight: 600,
            }}
          >
            {saving ? 'Saving…' : 'Save Changes'}
          </button>
        </div>
      </div>

      {/* Body */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Sidebar */}
        <div
          onDragOver={e => { e.preventDefault(); setDragOverCell('sidebar') }}
          onDragLeave={() => setDragOverCell(c => c === 'sidebar' ? null : c)}
          onDrop={e => { e.preventDefault(); dropOnSidebar() }}
          style={{
            width: 200, flexShrink: 0,
            background: sidebarIsOver ? 'rgba(30,58,95,0.4)' : 'rgba(9,9,11,0.97)',
            borderRight: `1px solid ${sidebarIsOver ? '#2563eb' : '#1c1c1e'}`,
            padding: 14, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 14,
            transition: 'background 0.1s, border-color 0.1s',
          }}
        >
          {/* Add artist */}
          <div>
            <div style={{
              fontSize: 10, fontWeight: 700, color: '#52525b',
              textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 8,
            }}>
              Add Artist
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <input
                value={newArtistName}
                onChange={e => setNewArtistName(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && addArtist()}
                placeholder="Name…"
                style={{
                  flex: 1, minWidth: 0,
                  background: '#111113', border: '1px solid #3f3f46', borderRadius: 6,
                  padding: '5px 8px', color: '#f4f4f5', fontSize: 12,
                  fontFamily: 'inherit', outline: 'none',
                }}
              />
              <button
                onClick={addArtist}
                style={{
                  background: '#1e3a5f', border: '1px solid #2563eb', borderRadius: 6,
                  color: '#60a5fa', fontSize: 14, fontWeight: 700,
                  cursor: 'pointer', padding: '0 10px', flexShrink: 0,
                }}
              >
                +
              </button>
            </div>
            <label
              style={{
                display: 'block', marginTop: 6, textAlign: 'center',
                background: '#18181b', border: '1px solid #3f3f46', borderRadius: 6,
                padding: '5px 0', color: '#a1a1aa', fontSize: 11, fontWeight: 500,
                cursor: 'pointer',
              }}
            >
              Upload .txt
              <input
                type="file" accept=".txt" onChange={handleFileUpload}
                style={{ display: 'none' }}
              />
            </label>
          </div>

          {/* Unassigned list */}
          <div style={{ flex: 1 }}>
            <div style={{
              fontSize: 10, fontWeight: 700, color: '#52525b',
              textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 8,
            }}>
              Unassigned ({unassigned.length})
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              {unassigned.map(artist => (
                <ArtistChip
                  key={artist}
                  artist={artist}
                  isDragging={drag?.artist === artist}
                  onDragStart={() => startDragFrom(artist, null, null)}
                  onDragEnd={() => setDrag(null)}
                  onRemove={() => removeArtist(artist)}
                />
              ))}
              {unassigned.length === 0 && (
                <div style={{ color: '#52525b', fontSize: 12, textAlign: 'center', paddingTop: 8 }}>
                  {sidebarIsOver ? 'Drop to unassign' : 'All assigned'}
                </div>
              )}
              {sidebarIsOver && unassigned.length > 0 && (
                <div style={{ color: '#60a5fa', fontSize: 12, textAlign: 'center', paddingTop: 4 }}>
                  Drop to unassign
                </div>
              )}
            </div>
          </div>

          {/* All artists (for removing assigned ones) */}
          {artists.filter(a => assignedSet.has(a)).length > 0 && (
            <div>
              <div style={{
                fontSize: 10, fontWeight: 700, color: '#52525b',
                textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 8,
              }}>
                Assigned ({artists.filter(a => assignedSet.has(a)).length})
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                {artists.filter(a => assignedSet.has(a)).map(artist => (
                  <div key={artist} style={{
                    display: 'flex', alignItems: 'center', gap: 5,
                    padding: '4px 8px', background: '#0c1a2e', border: '1px solid #1e3a5f',
                    borderRadius: 6, fontSize: 12, color: '#93c5fd',
                  }}>
                    <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {artist}
                    </span>
                    <button
                      onClick={() => removeArtist(artist)}
                      onMouseDown={e => e.stopPropagation()}
                      title="Remove artist"
                      style={{
                        background: 'none', border: 'none', color: '#7f1d1d',
                        cursor: 'pointer', fontSize: 14, lineHeight: 1,
                        padding: '0 2px', flexShrink: 0,
                      }}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Schedule grid */}
        <div style={{ flex: 1, overflowX: 'auto', overflowY: 'auto', padding: '16px 20px' }}>
          {stages.length === 0 ? (
            <div style={{ color: '#52525b', fontSize: 13, paddingTop: 24 }}>
              No stages found in venue GeoJSON.
            </div>
          ) : (
            <div style={{ display: 'flex', gap: 10, minWidth: stages.length * 190 }}>
              {stages.map(stage => (
                <div key={stage.id} style={{ flex: 1, minWidth: 180 }}>
                  <div style={{
                    textAlign: 'center', padding: '7px 0', marginBottom: 8,
                    borderBottom: `2px solid ${stage.color}`,
                    fontWeight: 800, fontSize: 12, color: stage.color,
                    textTransform: 'uppercase', letterSpacing: '0.1em',
                  }}>
                    {stage.name}
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {slots.map((slot, idx) => {
                      const cell = schedule[stage.id]?.[idx]
                      const artist = cell?.artist
                      const locked = cell?.locked
                      const manual = cell?.manual ?? true
                      const optimized = cell?.optimized ?? false
                      const isOptimized = artist && !locked && optimized
                      const isAuto = artist && !locked && !manual && !optimized
                      const cellKey = `${stage.id}:${idx}`
                      const isOver = dragOverCell === cellKey && !locked
                      let bg = '#0d0d0f', border = '#1c1c1e', borderStyle = 'solid'
                      if (locked) { bg = '#1a1200'; border = '#d97706' }
                      else if (isOver) { bg = '#0f1f3d'; border = '#60a5fa' }
                      else if (isOptimized) { bg = '#0d1530'; border = '#3b82f6'; borderStyle = 'dashed' }
                      else if (isAuto) { bg = '#0d1f1a'; border = '#16a34a'; borderStyle = 'dashed' }
                      else if (artist) { bg = '#0c1a2e'; border = '#2563eb' }
                      return (
                        <div
                          key={idx}
                          onDragOver={e => { e.preventDefault(); setDragOverCell(cellKey) }}
                          onDragLeave={() => setDragOverCell(c => c === cellKey ? null : c)}
                          onDrop={e => { e.preventDefault(); dropOnCell(stage.id, idx) }}
                          onDoubleClick={() => toggleLock(stage.id, idx)}
                          title={artist ? (locked ? 'Double-click to unlock' : `Double-click to lock · drag to move${isOptimized ? ' (optimizer-placed)' : isAuto ? ' (auto-filled)' : ''}`) : 'Drag artist here'}
                          style={{
                            border: `1px ${borderStyle} ${border}`, borderRadius: 6,
                            padding: '5px 8px', background: bg, minHeight: 52,
                            transition: 'border-color 0.1s, background 0.1s',
                            display: 'flex', flexDirection: 'column', gap: 3,
                          }}
                        >
                          <div style={{ fontSize: 10, color: '#3f3f46', fontVariantNumeric: 'tabular-nums', flexShrink: 0 }}>
                            {slot.label}
                          </div>
                          {artist ? (
                            <div
                              draggable={!locked}
                              onDragStart={e => {
                                if (locked) { e.preventDefault(); return }
                                e.stopPropagation()
                                startDragFrom(artist, stage.id, idx)
                              }}
                              onDragEnd={() => setDrag(null)}
                              style={{
                                display: 'flex', alignItems: 'center', gap: 5, minWidth: 0,
                                cursor: locked ? 'not-allowed' : 'grab',
                                opacity: drag?.artist === artist && drag?.fromStage === stage.id && drag?.fromSlot === idx ? 0.35 : 1,
                              }}
                            >
                              <span style={{
                                fontSize: 12, fontWeight: 600,
                                color: locked ? '#fcd34d' : isOptimized ? '#93c5fd' : isAuto ? '#86efac' : '#bfdbfe',
                                flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                                userSelect: 'none',
                              }}>
                                {artist}
                              </span>
                              {locked && <span style={{ fontSize: 11, flexShrink: 0 }}>🔒</span>}
                              {isOptimized && (
                                <span style={{
                                  fontSize: 8, fontWeight: 800, color: '#93c5fd',
                                  background: '#1e3a5f', borderRadius: 3, padding: '1px 3px',
                                  letterSpacing: '0.05em', flexShrink: 0,
                                }}>OPTIMAL</span>
                              )}
                              {isAuto && (
                                <span style={{
                                  fontSize: 8, fontWeight: 800, color: '#16a34a',
                                  background: '#052e16', borderRadius: 3, padding: '1px 3px',
                                  letterSpacing: '0.05em', flexShrink: 0,
                                }}>AUTO</span>
                              )}
                            </div>
                          ) : (
                            <div style={{ fontSize: 11, color: isOver ? '#60a5fa' : '#2a2a2e', fontStyle: 'italic' }}>
                              {isOver ? 'Drop here' : 'empty'}
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function ArtistChip({ artist, isDragging, onDragStart, onDragEnd, onRemove }) {
  const [hovered, setHovered] = useState(false)
  return (
    <div
      draggable
      onDragStart={e => { e.dataTransfer.effectAllowed = 'move'; onDragStart() }}
      onDragEnd={onDragEnd}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'flex', alignItems: 'center', gap: 5,
        padding: '5px 8px', background: '#27272a', border: '1px solid #3f3f46',
        borderRadius: 6, fontSize: 12, cursor: 'grab', color: '#e4e4e7',
        opacity: isDragging ? 0.35 : 1,
        userSelect: 'none',
      }}
    >
      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {artist}
      </span>
      <button
        onClick={e => { e.stopPropagation(); onRemove() }}
        onMouseDown={e => e.stopPropagation()}
        title="Remove artist"
        style={{
          background: 'none', border: 'none',
          color: hovered ? '#ef4444' : '#52525b',
          cursor: 'pointer', fontSize: 14, lineHeight: 1,
          padding: '0 2px', flexShrink: 0,
          transition: 'color 0.1s',
        }}
      >
        ×
      </button>
    </div>
  )
}

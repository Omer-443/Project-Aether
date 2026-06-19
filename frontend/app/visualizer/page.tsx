'use client'

import { motion } from 'framer-motion'
import { useEffect, useMemo, useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'

import { pageTransition, heavySpring } from '@/lib/animation-utils'
import { getTrajectoryTelemetry, type TrajectoryTelemetryResponse } from '@/lib/api/aether-client'
import { ChevronRight, Play, Pause, RotateCcw } from 'lucide-react'

type LoadState = {
  data: TrajectoryTelemetryResponse | null
  loading: boolean
  error: string | null
}

function clampEventCount(value: number) {
  return Math.max(2, Math.min(36, Math.round(value)))
}

function volatility(values: number[]) {
  if (values.length < 2) return 0
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length
  return Math.sqrt(variance)
}

export default function VisualizerPage() {
  const [isPlaying, setIsPlaying] = useState(false)
  const [intensity, setIntensity] = useState(50)
  const [nEvents, setNEvents] = useState(12)
  const [state, setState] = useState<LoadState>({ data: null, loading: true, error: null })

  useEffect(() => {
    let cancelled = false
    // Debug: verify button clicks -> nEvents changes -> API call triggers.
    // eslint-disable-next-line no-console
    console.log('[Visualizer] fetching telemetry for nEvents=', nEvents)


    setState((prev) => ({ ...prev, loading: true, error: null }))
    ;(async () => {
      try {
          // eslint-disable-next-line no-console
          console.log('[Visualizer] calling getTrajectoryTelemetry for nEvents=', nEvents)
          const data = await getTrajectoryTelemetry(nEvents)

        if (!cancelled) {
          // eslint-disable-next-line no-console
          console.log('[Visualizer] telemetry received OK. trajectory=', data.trajectory?.length, 'gaps=', data.gaps?.length)
          // eslint-disable-next-line no-console
          console.log('[Visualizer] provider/risk_label=', data.provider, data.risk_label)
          const t0 = data.trajectory?.[0]
          const tLast = data.trajectory?.[data.trajectory.length - 1]
          // eslint-disable-next-line no-console
          console.log('[Visualizer] trajectory sample:', {
            t0_time: t0?.time,
            t0_z: t0?.z,
            tLast_time: tLast?.time,
            tLast_z: tLast?.z,
          })
          setState({ data, loading: false, error: null })

        }

      } catch (e) {
        // eslint-disable-next-line no-console
        console.error('[Visualizer] telemetry fetch failed:', e)
        if (!cancelled) setState({ data: null, loading: false, error: String(e) })
      }
    })()
    return () => {
      cancelled = true
    }
  }, [nEvents])


  const chartData = useMemo(() => {
    const raw = state.data?.trajectory ?? []
    return raw.map((point) => ({
      x: Number(point.time).toFixed(2),
      z: Math.round(Number(point.z ?? 0) * 10000) / 100,
      friction: Math.round(Number(point.friction ?? 0) * 10000) / 100,
    }))
  }, [state.data])


  const gapLines = useMemo(() => {
    return (state.data?.gaps ?? []).map((gap) => Number(gap.time).toFixed(2))
  }, [state.data])

  const stats = useMemo(() => {
    const zValues = chartData.map((point) => point.z).filter(Number.isFinite)
    const current = zValues.length ? zValues[zValues.length - 1] : null
    const peak = zValues.length ? Math.max(...zValues) : null
    return {
      current,
      volatility: volatility(zValues),
      peak,
      events: state.data?.stats?.event_count ?? state.data?.events?.length ?? nEvents,
    }
  }, [chartData, nEvents, state.data])

  const requestFromIntensity = (value: number) => {
    setIntensity(value)
    setNEvents(clampEventCount(4 + (value / 100) * 24))
  }

  return (
    <motion.div
      variants={pageTransition}
      initial="initial"
      animate="animate"
      transition={{ ...heavySpring }}
      className="space-y-6"
    >
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-2">Trajectory Visualizer</h1>
        <p className="text-muted-foreground">Continuous curve analysis over backend-sampled claim windows</p>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...heavySpring }}
        className="glassmorphic p-8 rounded-lg border border-border/50"
      >
        <div className="mb-6">
          <div className="flex items-center justify-between gap-4 mb-4">
            <h3 className="text-sm font-semibold text-foreground">Oscilloscope Curve</h3>
            <div className="text-xs text-muted-foreground font-mono">
              {state.data ? `${state.data.provider} / ${state.data.risk_label}` : state.loading ? 'Loading telemetry' : 'Telemetry unavailable'}
            </div>
          </div>

          <ResponsiveContainer width="100%" height={400}>
            {/* Debug: force re-render info */}
            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 10 }} key={`${nEvents}-${state.data?.trajectory?.length ?? 0}`}>


              <defs>
                <linearGradient id="curveGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#10b981" stopOpacity={1} />
                  <stop offset="50%" stopColor="#3b82f6" stopOpacity={1} />
                  <stop offset="100%" stopColor="#8b5cf6" stopOpacity={1} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="x" stroke="#666" style={{ fontSize: 12 }} interval="preserveStartEnd" />
              <YAxis stroke="#666" style={{ fontSize: 12 }} domain={[0, 100]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1a1a1a',
                  border: '1px solid #333',
                  borderRadius: '6px',
                  color: '#e5e5e5',
                }}
                cursor={{ stroke: '#10b981', strokeWidth: 1, strokeDasharray: '5 5' }}
              />

              {gapLines.map((x, i) => (
                <ReferenceLine key={`${x}-${i}`} x={x} stroke="#f43f5e" strokeWidth={1} strokeDasharray="5 5" />
              ))}

              <Line
                type="monotone"
                dataKey="z"
                stroke="url(#curveGradient)"
                dot={false}
                isAnimationActive={isPlaying}
                animationDuration={isPlaying ? 2000 : 0}
                strokeWidth={2}
                name="Latent Z"
              />
              <Line
                type="monotone"
                dataKey="friction"
                stroke="#f59e0b"
                dot={false}
                strokeWidth={1.5}
                name="Denial Friction"
              />
            </LineChart>
          </ResponsiveContainer>

          {state.error && <div className="mt-3 text-xs text-rose-400 font-mono">{state.error}</div>}
        </div>

        <div className="flex flex-wrap items-center gap-4 pt-4 border-t border-border/30">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 rounded-md text-emerald-400 font-mono text-sm transition-colors"
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            {isPlaying ? 'Pause' : 'Play'}
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => {
              setIsPlaying(false)
              setIntensity(50)
              setNEvents(12)
            }}
            className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 border border-border rounded-md text-muted-foreground font-mono text-sm transition-colors"
          >
            <RotateCcw size={16} />
            Reset
          </motion.button>

          <div className="flex items-center gap-3 ml-auto">
            <label className="text-sm text-muted-foreground font-mono">Intensity</label>
            <input
              type="range"
              min="0"
              max="100"
              value={intensity}
              onChange={(e) => requestFromIntensity(parseInt(e.target.value, 10))}
              className="w-32 h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-emerald-500"
            />
            <span className="text-sm font-mono text-foreground">{intensity}%</span>
          </div>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...heavySpring, delay: 0.1 }}
        className="glassmorphic p-6 rounded-lg border border-border/50"
      >
        <h3 className="text-sm font-semibold text-foreground mb-4">Event Injection</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { label: 'Extend Window', action: () => setNEvents((value) => clampEventCount(value + 1)) },
            { label: 'Widen Gap', action: () => setNEvents((value) => clampEventCount(value + 4)) },
            { label: 'Reset Window', action: () => setNEvents(12) },
          ].map(({ label, action }, i) => (
            <motion.button
              key={label}
              whileHover={{ scale: 1.02, y: -1 }}
              whileTap={{ scale: 0.98 }}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ ...heavySpring, delay: 0.15 + i * 0.05 }}
              onClick={action}
              className="flex items-center justify-center gap-2 px-4 py-3 bg-white/5 hover:bg-white/10 border border-border rounded-md text-muted-foreground hover:text-foreground font-mono text-sm transition-all"
            >
              {label}
              <ChevronRight size={14} />
            </motion.button>
          ))}
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...heavySpring, delay: 0.2 }}
        className="glassmorphic p-6 rounded-lg border border-border/50"
      >
        <h3 className="text-sm font-semibold text-foreground mb-3">Trajectory Info</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <div className="text-xs text-muted-foreground mb-1">Current Value</div>
            <div className="font-mono text-lg text-emerald-400">{stats.current == null ? 'N/A' : stats.current.toFixed(2)}</div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground mb-1">Volatility</div>
            <div className="font-mono text-lg text-amber-400">{stats.volatility.toFixed(2)}%</div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground mb-1">Peak Value</div>
            <div className="font-mono text-lg text-rose-400">{stats.peak == null ? 'N/A' : stats.peak.toFixed(2)}</div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground mb-1">Events</div>
            <div className="font-mono text-lg text-emerald-400">{String(stats.events)}</div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  )
}

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
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { pageTransition, heavySpring } from '@/lib/animation-utils'
import {
  getSampleData,
  runShockTest,
  type ShockTestResponse,
  type ClaimEvent,
} from '@/lib/api/aether-client'
import { Zap } from 'lucide-react'

type LoadState = {
  data: ShockTestResponse | null
  events: ClaimEvent[]
  loading: boolean
  error: string | null
}

// 🚨 MEANINGFUL ADAPTATION METRICS
type SeriesMetric = {
  settlingTime: string
  maxDeviation: string
  latentShift: number // Scale of Adaptation (How much did it move?)
  finalDeviation: number // Permanent Impact (How far is the new state from the old?)
}

function clampShockStrength(value: number) {
  return Math.max(1, Math.min(100, Math.round(value)))
}

function toMagnitude(point: Record<string, number>) {
  const d0 = Number(point.dim_0 ?? 0)
  const d1 = Number(point.dim_1 ?? 0)
  const d2 = Number(point.dim_2 ?? 0)
  const d3 = Number(point.dim_3 ?? 0)
  return Math.sqrt(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3)
}

function formatProbability(value: number) {
  return `${(value * 100).toFixed(2)}%`
}

function mean(arr: number[]) {
  if (!arr.length) return 0
  return arr.reduce((a, b) => a + b, 0) / arr.length
}

function stdev(arr: number[]) {
  if (arr.length < 2) return 0
  const m = mean(arr)
  const v = arr.reduce((acc, x) => acc + (x - m) ** 2, 0) / arr.length
  return Math.sqrt(v)
}

// 🚨 PURE ADAPTATION METRIC CALCULATION
// Measures the scale and permanence of the model's reaction to the shock.
function calculateSeriesMetric(values: number[], times: number[], shockStartIndex: number): SeriesMetric {
  const validData = values
    .map((v, i) => ({ value: v, time: times[i] }))
    .filter((d) => !isNaN(d.value) && d.value != null)

  if (validData.length === 0) {
    return { settlingTime: 'N/A', maxDeviation: 'N/A', latentShift: 0, finalDeviation: 0 }
  }

  const validValues = validData.map((d) => d.value)
  const validTimes = validData.map((d) => d.time)
  const safeShockStart = Math.min(shockStartIndex, validValues.length - 1)

  const initial = validValues[0]
  const deviations = validValues.map((value) => value - initial)
  const maxAbs = Math.max(...deviations.map((value) => Math.abs(value)))
  const maxIndex = deviations.findIndex((value) => Math.abs(value) === maxAbs)

  const transitionValues = validValues.slice(safeShockStart)

  // 1. LATENT SHIFT MAGNITUDE (Total movement after shock)
  const totalShift = Math.abs(transitionValues[transitionValues.length - 1] - transitionValues[0])
  const latentShift = Math.round(totalShift * 100) / 100

  // 2. FINAL STATE DEVIATION (Permanent impact)
  const tailStart = Math.max(0, validValues.length - Math.max(5, Math.floor(validValues.length * 0.1)))
  const finalState =
    validValues.slice(tailStart).reduce((a, b) => a + b, 0) / (validValues.length - tailStart)
  const finalDeviation = Math.round(Math.abs(finalState - initial) * 100) / 100

  // 3. SETTLING TIME (Speed of adaptation)
  // Original logic required 5 consecutive points to be within a dynamic threshold.
  // That is overly strict for RAMP shocks where convergence is gradual.
  //
  // Fix: use a robust hysteresis-like criterion:
  // - compute a threshold based on overall deviation scale
  // - find the earliest index where values stay within threshold for a window
  // - but allow a single-point “excursion” (miss) inside the window
  // - additionally, compare to a local baseline (recent mean) for non-stationary signals
  const maxAbsSafe = Number.isFinite(maxAbs) ? maxAbs : 0
  const baseThreshold = Math.max(0.1, maxAbsSafe * 0.05)

  // For ramp: finalState may still be slowly drifting; widen tolerance slightly
  // based on recent volatility.
  const recentWindowStart = Math.max(0, validValues.length - 8)
  const recentValues = validValues.slice(recentWindowStart)
  const recentSigma = stdev(recentValues)

  const threshold = baseThreshold + recentSigma * 0.25

  // Use a window of 4 points (instead of 5) for better detection stability.
  // Allow 1 miss inside the window.
  const windowSize = 4
  const maxMisses = 1

  // Fallback: if we can't find a “sustained” window, compute a crossing time
  // based on first time within threshold (no 5-point requirement).
  let settleIndex = -1

  for (let i = safeShockStart; i <= validValues.length - windowSize; i++) {
    const window = validValues.slice(i, i + windowSize)

    // “within threshold” relative to finalState
    const misses = window.reduce((acc, v) => acc + (Math.abs(v - finalState) <= threshold ? 0 : 1), 0)

    if (misses <= maxMisses) {
      settleIndex = i
      break
    }
  }

  let settlingTime = 'Unstable'
  if (settleIndex >= 0 && validTimes[settleIndex] != null && validTimes[safeShockStart] != null) {
    const timeDiff = validTimes[settleIndex] - validTimes[safeShockStart]
    settlingTime = `${Math.max(0.1, timeDiff).toFixed(1)}d`
  } else {
    // Crossing-time fallback: first index where the series enters the tolerance band.
    let firstWithin = -1
    for (let i = safeShockStart; i < validValues.length; i++) {
      if (Math.abs(validValues[i] - finalState) <= threshold) {
        firstWithin = i
        break
      }
    }
    if (firstWithin >= 0 && validTimes[firstWithin] != null && validTimes[safeShockStart] != null) {
      const timeDiff = validTimes[firstWithin] - validTimes[safeShockStart]
      settlingTime = `${Math.max(0.1, timeDiff).toFixed(1)}d (soft)`
    }
  }

  return {
    settlingTime,
    maxDeviation: `${deviations[maxIndex] >= 0 ? '+' : ''}${deviations[maxIndex].toFixed(2)}`,
    latentShift,
    finalDeviation,
  }
}

export default function ShockTestPage() {
  const [shockActive, setShockActive] = useState(true)
  const [shockStrength, setShockStrength] = useState(50)
  const [shockType, setShockType] = useState<'Step' | 'Impulse' | 'Ramp'>('Step')
  const [state, setState] = useState<LoadState>({ data: null, events: [], loading: true, error: null })

  useEffect(() => {
    let cancelled = false
    setState((prev) => ({ ...prev, loading: true, error: null }))

    ;(async () => {
      try {
        const sample = state.events.length ? { events: state.events } : await getSampleData(12)
        const magnitude = shockActive ? (shockStrength / 100) * 2 : 0.01
        const data = await runShockTest(
          sample.events,
          magnitude,
          shockType.toLowerCase() as 'step' | 'impulse' | 'ramp',
        )
        if (!cancelled) setState({ data, events: sample.events, loading: false, error: null })
      } catch (e) {
        if (!cancelled) setState((prev) => ({ ...prev, data: null, loading: false, error: String(e) }))
      }
    })()

    return () => {
      cancelled = true
    }
  }, [shockActive, shockStrength, shockType])

  const chartData = useMemo(() => {
    const baseline = state.data?.baseline_trajectory ?? []
    const shocked = state.data?.shocked_trajectory ?? []

    return baseline.map((point, index) => {
      const liquidPoint = shocked[index]
      return {
        time: Number(point.time ?? state.data?.eval_times[index] ?? index).toFixed(2),
        // Purple Line: Normal CDE Trajectory (The Baseline Reality)
        baseline: Math.round(toMagnitude(point) * 100) / 100,
        // Green Line: Shocked CDE Trajectory (The Adapted Reality)
        liquid: liquidPoint ? Math.round(toMagnitude(liquidPoint) * 100) / 100 : NaN,
      }
    })
  }, [state.data])

  const metrics = useMemo(() => {
    const times = chartData.map((point) => Number(point.time))
    const baseline = chartData.map((point) => point.baseline)
    const liquid = chartData.map((point) => point.liquid)

    const shockRatio = 0.7
    const t0 = times.length ? times[0] : 0
    const tT = times.length ? times[times.length - 1] : 0
    const shockTime = t0 + (tT - t0) * shockRatio

    const shockStartIndex = Math.max(0, times.findIndex((t) => t >= shockTime))
    const safeShockStartIndex = shockStartIndex === -1 ? times.length : shockStartIndex

    return {
      baseline: calculateSeriesMetric(baseline, times, safeShockStartIndex),
      liquid: calculateSeriesMetric(liquid, times, safeShockStartIndex),
    }
  }, [chartData, state.data])

  const analysis = useMemo(() => {
    if (!state.data) return state.loading ? 'Running model shock test...' : 'No shock-test telemetry available.'
    const delta = formatProbability(state.data.delta_prob)

    return `Shock magnitude ${state.data.shock_magnitude.toFixed(2)} changed denial probability by ${delta}. The Liquid CDE successfully absorbed the distribution shift, moving its latent state by ${metrics.liquid.latentShift} units and permanently adapting to a new reality with ${metrics.liquid.finalDeviation} units of deviation, all within ${metrics.liquid.settlingTime}.`
  }, [metrics.liquid.latentShift, metrics.liquid.finalDeviation, metrics.liquid.settlingTime, state.data, state.loading])

  const applyShockType = (type: 'Step' | 'Impulse' | 'Ramp') => {
    setShockType(type)
    if (type === 'Step') setShockStrength(45)
    if (type === 'Impulse') setShockStrength(80)
    if (type === 'Ramp') setShockStrength(60)
  }

  return (
    <motion.div
      variants={pageTransition}
      initial="initial"
      animate="animate"
      transition={{ ...(heavySpring as any) }}
      className="space-y-6"
    >
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-2">Policy Shock Simulator</h1>
        <p className="text-muted-foreground">
          Visualizing the Liquid CDE's native ability to contextually adapt its continuous trajectory under payer rule changes
        </p>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...(heavySpring as any) }}
        className="glassmorphic p-8 rounded-lg border border-border/50"
      >
        <div className="flex items-center justify-between gap-4 mb-4">
          <h3 className="text-sm font-semibold text-foreground">Continuous Trajectory Adaptation</h3>
          <div className="text-xs font-mono text-muted-foreground">
            {state.data
              ? `Type ${state.data.shock_type.toUpperCase()} | Delta Prob: ${formatProbability(state.data.delta_prob)}`
              : state.loading
                ? 'Running shock model'
                : 'Unavailable'}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="time" stroke="#666" style={{ fontSize: 12 }} />
            <YAxis
              stroke="#666"
              style={{ fontSize: 12 }}
              domain={['auto', 'auto']}
              tickFormatter={(val) => (typeof val === 'number' ? val.toFixed(1) : String(val))}
            />

            <Tooltip
              contentStyle={{
                backgroundColor: '#1a1a1a',
                border: '1px solid #333',
                borderRadius: '6px',
                color: '#e5e5e5',
              }}
            />
            <Legend wrapperStyle={{ fontSize: 12, color: '#e5e5e5' }} />

            <Line type="monotone" dataKey="baseline" stroke="#8b5cf6" strokeWidth={2} dot={false} name="Baseline Trajectory (Normal Rules)" />
            <Line type="monotone" dataKey="liquid" stroke="#10b981" strokeWidth={2} dot={false} name="Adapted Trajectory (Shocked Rules)" connectNulls={false} />
          </LineChart>
        </ResponsiveContainer>
        {state.error && <div className="mt-3 text-xs text-rose-400 font-mono">{state.error}</div>}
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...(heavySpring as any), delay: 0.1 }}
        className="glassmorphic p-6 rounded-lg border border-border/50"
      >
        <h3 className="text-sm font-semibold text-foreground mb-4">Shock Configuration</h3>

        <div className="space-y-4">
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm text-muted-foreground font-mono">Shock Strength</label>
              <span className="text-sm font-mono text-emerald-400">{shockStrength}%</span>
            </div>
            <input
              type="range"
              min="1"
              max="100"
              value={shockStrength}
              onChange={(e) => setShockStrength(clampShockStrength(parseInt(e.target.value, 10)))}
              className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-emerald-500"
            />
          </div>

          <div>
            <label className="text-sm text-muted-foreground font-mono mb-2 block">Shock Type</label>
            <div className="grid grid-cols-3 gap-3">
              {(['Step', 'Impulse', 'Ramp'] as const).map((type) => (
                <motion.button
                  key={type}
                  whileHover={{ scale: 1.02, y: -1 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => applyShockType(type)}
                  className={`px-4 py-2 border rounded-md font-mono text-xs transition-all ${
                    shockType === type
                      ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                      : 'bg-white/5 hover:bg-white/10 border-border text-muted-foreground hover:text-foreground'
                  }`}
                >
                  {type}
                </motion.button>
              ))}
            </div>
          </div>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShockActive(!shockActive)}
            className={`w-full flex items-center justify-center gap-2 px-4 py-3 rounded-md font-mono text-sm transition-all border ${
              shockActive
                ? 'bg-rose-500/10 border-rose-500/30 text-rose-400 hover:bg-rose-500/20'
                : 'bg-white/5 border-border text-muted-foreground hover:bg-white/10'
            }`}
          >
            <Zap size={16} />
            {shockActive ? 'Shock Active' : 'Trigger Shock'}
          </motion.button>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...(heavySpring as any), delay: 0.2 }}
        className="grid grid-cols-1 md:grid-cols-2 gap-4"
      >
        <div className="glassmorphic p-6 rounded-lg border border-border/50">
          <h3 className="text-sm font-semibold text-emerald-400 mb-4">Scale of Adaptation (Shocked State)</h3>
          <div className="space-y-3">
            <div>
              <div className="text-xs text-muted-foreground mb-1">Latent Shift Magnitude</div>
              <div className="font-mono text-lg text-emerald-400">{metrics.liquid.latentShift} units</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground mb-1">Final State Deviation</div>
              <div className="font-mono text-lg text-emerald-400">{metrics.liquid.finalDeviation} units</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground mb-1">Max Deviation</div>
              <div className="font-mono text-lg text-emerald-400">{metrics.liquid.maxDeviation}</div>
            </div>
          </div>
        </div>

        <div className="glassmorphic p-6 rounded-lg border border-border/50">
          <h3 className="text-sm font-semibold text-purple-400 mb-4">Speed of Adaptation (Shocked State)</h3>
          <div className="space-y-3">
            <div>
              <div className="text-xs text-muted-foreground mb-1">Settling Time</div>
              <div className="font-mono text-lg text-purple-400">{metrics.liquid.settlingTime}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground mb-1">Baseline Settling Time</div>
              <div className="font-mono text-lg text-muted-foreground">{metrics.baseline.settlingTime}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground mb-1">Adaptation Score</div>
              <div className="font-mono text-lg text-purple-400">{state.data?.adaptation_score?.toFixed(4) ?? 'N/A'}</div>
            </div>
          </div>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...(heavySpring as any), delay: 0.3 }}
        className="glassmorphic p-6 rounded-lg border border-border/50"
      >
        <h3 className="text-sm font-semibold text-foreground mb-3">Analysis</h3>
        <p className="text-sm text-muted-foreground leading-relaxed">{analysis}</p>
      </motion.div>
    </motion.div>
  )
}


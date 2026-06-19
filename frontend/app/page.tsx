'use client'

import { motion } from 'framer-motion'
import { Activity, TrendingUp, AlertCircle, Clock } from 'lucide-react'
import { MetricCard } from '@/components/MetricCard'
import { pageTransition, staggerContainer, heavySpring } from '@/lib/animation-utils'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts'

import { useEffect, useMemo, useState } from 'react'
import { getDashboardTelemetry, type DashboardMetric } from '@/lib/api/aether-client'

type LoadState<T> = { data: T | null; loading: boolean; error: string | null }

type DriftRow = {
  id: string
  policy: string
  drift: number | null
  severity: 'high' | 'medium' | 'low'
}


export default function Dashboard() {
  const [state, setState] = useState<LoadState<Awaited<ReturnType<typeof getDashboardTelemetry>>>>({
    data: null,
    loading: true,
    error: null,
  })

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const data = await getDashboardTelemetry()
        if (!cancelled) setState({ data, loading: false, error: null })
      } catch (e) {
        if (!cancelled) setState({ data: null, loading: false, error: String(e) })
      }
    }
    load()
    const interval = window.setInterval(load, 30000)
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [])

  const ingestionChart = useMemo(() => {
    const ingestion = state.data?.ingestion ?? []
    // Use latency_ms directly for charting; avoids inventing scale factors.
    const values = ingestion.map((p) => p.latency_ms).filter((v): v is number => Number.isFinite(v))
    const min = values.length ? Math.min(...values) : 0
    const max = values.length ? Math.max(...values) : 1

    // Normalize to 0..100 for a consistent UI scale.
    const denom = max - min || 1
    return ingestion.map((p) => {
      const t = Number.isFinite(p.t) ? p.t : 0
      const time = t.toString().padStart(2, '0') + ':00'
      const raw = Number.isFinite(p.latency_ms) ? p.latency_ms : 0
      const normalized = ((raw - min) / denom) * 100
      return {
        time,
        value: Math.max(0, Math.min(100, Math.round(normalized))),
      }
    })
  }, [state.data])

  const driftRows = useMemo(() => {
    const drifts = state.data?.drifts ?? []
    return drifts.map((d) => ({
      id: d.id,
      policy: d.id,
      drift: Number.isFinite(d.drift_score) ? Number(d.drift_score) : null,
      severity: d.severity,
    }))
  }, [state.data])

  return (
    <motion.div
      variants={pageTransition}
      initial="initial"
      animate="animate"
      transition={{ ...heavySpring }}
      className="space-y-8"
    >
      {/* Page Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-2">Dashboard</h1>
        <p className="text-muted-foreground">Real-time telemetry and system health overview</p>
        {state.data?.status ? (
          <div className="mt-3 flex flex-wrap gap-2 text-xs font-mono">
            <span className="rounded-full border border-border bg-white/5 px-3 py-1 text-muted-foreground">
              Source: {String(state.data.status.environment ?? 'unknown')}
            </span>
            <span className="rounded-full border border-border bg-white/5 px-3 py-1 text-muted-foreground">
              Model: {String(state.data.status.model ?? 'unknown')}
            </span>
            <span className="rounded-full border border-border bg-white/5 px-3 py-1 text-muted-foreground">
              Checkpoint: {Boolean(state.data.status.model_loaded_from_checkpoint) ? 'loaded' : 'not loaded'}
            </span>
          </div>
        ) : null}
      </div>

      {/* Metrics Grid */}
      <motion.div
        variants={staggerContainer}
        initial="initial"
        animate="animate"
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"
      >
        {state.loading || !state.data ? (
          Array.from({ length: 4 }).map((_, i) => (
            <MetricCard
              key={i}
              label="Loading"
              value="…"
              change=""
              changeType="neutral"
              icon={<Activity size={20} className="text-muted-foreground" />}
            />
          ))
        ) : (
          state.data.metrics.slice(0, 4).map((m: DashboardMetric, idx: number) => {
            const icon = idx === 0 ? (
              <Activity size={20} className={m.tone === 'emerald' ? 'text-emerald-400' : m.tone === 'rose' ? 'text-rose-400' : 'text-amber-400'} />
            ) : idx === 1 ? (
              <TrendingUp size={20} className={m.tone === 'emerald' ? 'text-emerald-400' : m.tone === 'rose' ? 'text-rose-400' : 'text-amber-400'} />
            ) : idx === 2 ? (
              <AlertCircle size={20} className={m.tone === 'emerald' ? 'text-emerald-400' : m.tone === 'rose' ? 'text-rose-400' : 'text-amber-400'} />
            ) : (
              <Clock size={20} className={m.tone === 'emerald' ? 'text-emerald-400' : m.tone === 'rose' ? 'text-rose-400' : 'text-amber-400'} />
            )

            const changeType = m.tone === 'emerald' ? 'positive' : m.tone === 'rose' ? 'negative' : 'neutral'

            return (
              <MetricCard
                key={m.label}
                label={m.label}
                value={m.value}
                change={m.delta}
                changeType={changeType}
                icon={icon}
              >
                <div className="text-xs text-muted-foreground font-mono mt-3">{m.description}</div>
              </MetricCard>
            )
          })
        )}
      </motion.div>


      {/* Charts Section */}
      <motion.div
        variants={staggerContainer}
        initial="initial"
        animate="animate"
        className="grid grid-cols-1 lg:grid-cols-3 gap-4"
      >
        {/* Health Trend */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ ...heavySpring }}
          className="lg:col-span-2 glassmorphic p-6 rounded-lg border border-border/50 hover:border-emerald-500/20 transition-all"
        >
          <h3 className="text-sm font-semibold text-foreground mb-4">System Health Trend</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={ingestionChart}>
              <defs>

                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="time" stroke="#666" style={{ fontSize: 12 }} />
              <YAxis stroke="#666" style={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1a1a1a',
                  border: '1px solid #333',
                  borderRadius: '6px',
                  color: '#e5e5e5',
                }}
              />
              <Area
                type="monotone"
                dataKey="value"
                stroke="#10b981"
                fillOpacity={1}
                fill="url(#colorValue)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Quick Stats */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ ...heavySpring, delay: 0.1 }}
          className="glassmorphic p-6 rounded-lg border border-border/50"
        >
          <h3 className="text-sm font-semibold text-foreground mb-4">Last 24h</h3>
          <div className="space-y-3">
            <div>
              <div className="text-xs text-muted-foreground mb-1">Avg Response Time</div>
              <div className="font-mono text-lg text-emerald-400">
                {state.data?.ingestion?.length
                  ? `${Math.round(
                      state.data.ingestion.reduce((a, p) => a + (Number.isFinite(p.latency_ms) ? p.latency_ms : 0), 0) /
                        state.data.ingestion.length
                    )}ms`
                  : '—'}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground mb-1">Denied Rate</div>
              <div className="font-mono text-lg text-emerald-400">
                {state.data?.ingestion?.length
                  ? (() => {
                      const total = state.data.ingestion.reduce((a, p) => a + (Number.isFinite(p.claims) ? p.claims : 0), 0)
                      const denied = state.data.ingestion.reduce((a, p) => a + (Number.isFinite(p.denied) ? p.denied : 0), 0)
                      const pct = total ? (denied / total) * 100 : 0
                      return `${pct.toFixed(2)}%`
                    })()
                  : '—'}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground mb-1">Throughput</div>
              <div className="font-mono text-lg text-emerald-400">
                {state.data?.ingestion?.length
                  ? (() => {
                      const totalClaims = state.data.ingestion.reduce((a, p) => a + (Number.isFinite(p.claims) ? p.claims : 0), 0)
                      const totalSeconds = 24 * 60 * 60
                      return `${(totalClaims / totalSeconds).toFixed(2)} claims/sec`
                    })()
                  : '—'}
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>

      {/* Policy Drift Table */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...heavySpring, delay: 0.2 }}
        className="glassmorphic p-6 rounded-lg border border-border/50"
      >
        <h3 className="text-sm font-semibold text-foreground mb-4">Recent Policy Drifts</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm font-mono">
            <thead>
              <tr className="border-b border-border/30">
                <th className="text-left py-2 px-3 text-muted-foreground font-normal">Policy</th>
                <th className="text-left py-2 px-3 text-muted-foreground font-normal">Drift</th>
                <th className="text-left py-2 px-3 text-muted-foreground font-normal">Status</th>
              </tr>
            </thead>
            <tbody>
              {driftRows.length ? driftRows.map((row) => (
                <tr key={row.id} className="border-b border-border/20 hover:bg-white/[0.02] transition-colors">
                  <td className="py-2 px-3 text-foreground">{row.policy}</td>
                  <td className="py-2 px-3 text-emerald-400">{row.drift == null ? 'N/A' : row.drift.toFixed(4)}</td>
                  <td className="py-2 px-3">
                    {row.severity === 'high' && <span className="text-rose-400">●</span>}
                    {row.severity === 'medium' && <span className="text-amber-400">●</span>}
                    {row.severity === 'low' && <span className="text-emerald-400">●</span>}
                  </td>
                </tr>
              )) : (
                <tr>
                  <td colSpan={3} className="py-6 px-3 text-center text-muted-foreground">No drift telemetry available</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </motion.div>
    </motion.div>
  )
}

'use client'

import { motion } from 'framer-motion'
import { useEffect, useMemo, useState } from 'react'
import { pageTransition, heavySpring, staggerContainer } from '@/lib/animation-utils'
import { getLogsTelemetry, type LogsRow } from '@/lib/api/aether-client'
import { Search, Filter } from 'lucide-react'

export default function LogsPage() {
  const [searchTerm, setSearchTerm] = useState('')
  const [filterLevel, setFilterLevel] = useState<'ALL' | LogsRow['status']>('ALL')
  const [loading, setLoading] = useState(true)

  type DisplayLog = {
    id: string
    time: string
    level: LogsRow['status']
    module: string
    action: string
    friction: number
  }

  const [logs, setLogs] = useState<DisplayLog[]>([])

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        setLoading(true)
        const res = await getLogsTelemetry(80)
        if (cancelled) return

        const mapped: DisplayLog[] = res.rows.map((r: LogsRow) => ({
          id: r.id,
          time: r.timestamp,
          level: r.status,
          module: `CLAIM_${r.status}`,
          action: r.cpt_opaque_id,
          friction: Number(r.friction_score),
        }))

        setLogs(mapped)
      } catch {
        if (!cancelled) setLogs([])
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    const interval = window.setInterval(load, 30000)

    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [])

  const filteredLogs = useMemo(() => {
    return logs.filter((log) => {
      const matchesSearch = log.action.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesLevel = filterLevel === 'ALL' || log.level === filterLevel
      return matchesSearch && matchesLevel
    })
  }, [logs, searchTerm, filterLevel])

  const getLevelColor = (level: DisplayLog['level']) => {
    switch (level) {
      case 'DENIED':
        return 'text-rose-400'
      case 'PENDING':
        return 'text-amber-400'
      case 'APPROVED':
        return 'text-emerald-400'
      case 'APPEALED':
        return 'text-sky-400'
      default:
        return 'text-foreground'
    }
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
        <h1 className="text-3xl font-bold text-foreground mb-2">Raw Telemetry Logs</h1>
        <p className="text-muted-foreground">Dense monospace data for all system events</p>
      </div>

      <motion.div
        variants={staggerContainer}
        initial="initial"
        animate="animate"
        className="glassmorphic p-4 rounded-lg border border-border/50 space-y-4"
      >
        <div className="flex items-center gap-2 bg-white/5 border border-border rounded-md px-3 py-2">
          <Search size={16} className="text-muted-foreground" />
          <input
            type="text"
            placeholder="Search logs..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="flex-1 bg-transparent text-foreground font-mono text-sm outline-none placeholder:text-muted-foreground"
          />
        </div>

        <div className="flex flex-wrap gap-2 items-center">
          <Filter size={16} className="text-muted-foreground" />
          {(['ALL', 'APPROVED', 'DENIED', 'PENDING', 'APPEALED'] as const).map((level) => (
            <motion.button
              key={level}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setFilterLevel(level)}
              className={`px-3 py-1 rounded-md font-mono text-xs transition-all border ${
                filterLevel === level
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                  : 'bg-white/5 border-border text-muted-foreground hover:bg-white/10'
              }`}
            >
              {level}
            </motion.button>
          ))}
        </div>

        <div className="text-xs text-muted-foreground font-mono">
          Showing {filteredLogs.length} of {logs.length} events
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...heavySpring }}
        className="glassmorphic p-0 rounded-lg border border-border/50 overflow-hidden"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead className="border-b border-border/30 bg-white/[0.02]">
              <tr>
                <th className="text-left py-3 px-4 text-muted-foreground font-normal">Time</th>
                <th className="text-left py-3 px-4 text-muted-foreground font-normal">Level</th>
                <th className="text-left py-3 px-4 text-muted-foreground font-normal">Module</th>
                <th className="text-left py-3 px-4 text-muted-foreground font-normal">Event</th>
                <th className="text-right py-3 px-4 text-muted-foreground font-normal">Friction</th>
              </tr>
            </thead>
            <tbody>
              {filteredLogs.length > 0 ? (
                filteredLogs.map((log, idx) => (
                  <motion.tr
                    key={log.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ ...heavySpring, delay: idx * 0.01 }}
                    className="border-b border-border/20 hover:bg-white/[0.02] transition-colors"
                  >
                    <td className="py-2 px-4 text-muted-foreground">{log.time}</td>
                    <td className={`py-2 px-4 font-bold ${getLevelColor(log.level)}`}>
                      {log.level}
                    </td>
                    <td className="py-2 px-4 text-foreground">{log.module}</td>
                    <td className="py-2 px-4 text-foreground">{log.action}</td>
                    <td className="py-2 px-4 text-right text-emerald-400">{log.friction.toFixed(3)}</td>
                  </motion.tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="py-8 px-4 text-center text-muted-foreground">
                    No logs matching criteria
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="border-t border-border/30 bg-white/[0.02] px-4 py-3 text-xs text-muted-foreground font-mono">
          End of logs • Latest entries shown first
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...heavySpring, delay: 0.1 }}
        className="grid grid-cols-2 md:grid-cols-4 gap-4"
      >
        {[
          { label: 'Total Events', value: logs.length },
          { label: 'Denied', value: logs.filter((l) => l.level === 'DENIED').length },
          { label: 'Pending', value: logs.filter((l) => l.level === 'PENDING').length },
          {
            label: 'Avg Friction',
            value: `${logs.length ? (logs.reduce((sum, log) => sum + (Number.isFinite(log.friction) ? log.friction : 0), 0) / logs.length).toFixed(3) : '0.000'}`,
          },
        ].map((stat) => (
          <div key={stat.label} className="glassmorphic p-4 rounded-lg border border-border/50">
            <div className="text-xs text-muted-foreground mb-1 font-mono">{stat.label}</div>
            <div className="text-lg font-mono font-bold text-emerald-400">{stat.value}</div>
          </div>
        ))}
      </motion.div>
    </motion.div>
  )
}


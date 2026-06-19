'use client'

import { motion } from 'framer-motion'
import { ReactNode } from 'react'

interface MetricCardProps {
  label: string
  value: string | number
  change?: string
  changeType?: 'positive' | 'negative' | 'neutral'
  icon?: ReactNode
  children?: ReactNode
}

export function MetricCard({
  label,
  value,
  change,
  changeType = 'neutral',
  icon,
  children,
}: MetricCardProps) {
  const changeColor = {
    positive: 'text-emerald-400',
    negative: 'text-rose-400',
    neutral: 'text-muted-foreground',
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', stiffness: 100, damping: 15 }}
      whileHover={{ y: -2 }}
      className="glassmorphic p-6 rounded-lg border border-border/50 hover:border-emerald-500/20 transition-all duration-300"
    >
      {/* Header with icon */}
      {icon && (
        <div className="flex items-center justify-between mb-4">
          <div className="p-2 bg-emerald-500/10 rounded-md border border-emerald-500/10">
            {icon}
          </div>
        </div>
      )}

      {/* Label */}
      <div className="metric-label mb-2">{label}</div>

      {/* Value */}
      <div className="metric-value mb-3">{value}</div>

      {/* Change indicator */}
      {change && (
        <div className={`text-sm font-mono ${changeColor[changeType]}`}>{change}</div>
      )}

      {/* Custom content */}
      {children}
    </motion.div>
  )
}

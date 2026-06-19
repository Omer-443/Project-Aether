'use client'

import { motion } from 'framer-motion'
import { Activity, Zap } from 'lucide-react'

export function TopBar() {
  const apiDocsUrl = `${process.env.NEXT_PUBLIC_AETHER_API_BASE_URL ?? 'http://localhost:8000'}/docs`

  return (
    <motion.header
      initial={{ y: -60 }}
      animate={{ y: 0 }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      className="fixed top-0 right-0 left-0 h-15 bg-background border-b border-border/50 glassmorphic flex items-center justify-between px-6"
      style={{ zIndex: 50 }}
    >
      {/* Left: Branding */}
      <div className="flex items-center gap-3">
        <div className="p-2 bg-emerald-500/10 rounded-md border border-emerald-500/20">
          <Zap size={20} className="text-emerald-400" />
        </div>
        <div className="flex flex-col">
          <h1 className="text-sm font-bold text-foreground">Project Aether</h1>
          <p className="text-xs text-muted-foreground">Enterprise Telemetry</p>
        </div>
      </div>

      {/* Right: Controls */}
      <div className="flex items-center gap-4">
        <motion.a
          href={apiDocsUrl}
          target="_blank"
          rel="noreferrer"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="p-2 hover:bg-white/5 rounded-md transition-colors"
          aria-label="Open API docs"
        >
          <Activity size={18} className="text-muted-foreground hover:text-foreground transition-colors" />
        </motion.a>
      </div>
    </motion.header>
  )
}

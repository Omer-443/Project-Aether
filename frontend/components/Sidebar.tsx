'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useState } from 'react'
import { Menu, X, BarChart3, Zap, RefreshCw, FileText } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import pkg from '@/package.json'

export function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const pathname = usePathname()

  const routes = [
    { href: '/', label: 'Dashboard', icon: BarChart3 },
    { href: '/visualizer', label: 'Trajectory', icon: Zap },
    { href: '/shock-test', label: 'Shock Test', icon: RefreshCw },
    { href: '/logs', label: 'Telemetry', icon: FileText },
  ]

  const isActive = (href: string) => pathname === href

  return (
    <motion.aside
      id="aether-sidebar"
      animate={{ width: isCollapsed ? 64 : 240 }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      className="fixed left-0 top-0 h-screen bg-background border-r border-border flex flex-col"
      style={{ zIndex: 40, paddingTop: 60 }}
    >
      {/* Toggle Button */}
      <div className="flex items-center justify-center h-16 border-b border-border/50">
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="p-2 hover:bg-white/5 rounded-md transition-colors"
          aria-label="Toggle sidebar"
        >
          {isCollapsed ? (
            <Menu size={20} className="text-muted-foreground" />
          ) : (
            <X size={20} className="text-muted-foreground" />
          )}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-hidden">
        <div className="space-y-1 p-3">
          {routes.map(({ href, label, icon: Icon }) => {
            const active = isActive(href)
            return (
              <Link key={href} href={href}>
                <motion.div
                  whileHover={{ x: 2 }}
                  whileTap={{ scale: 0.98 }}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-md transition-all cursor-pointer group ${
                    active
                      ? 'bg-emerald-500/10 border border-emerald-500/20'
                      : 'hover:bg-white/5 border border-transparent'
                  }`}
                >
                  <Icon
                    size={20}
                    className={`transition-colors ${
                      active ? 'text-emerald-400' : 'text-muted-foreground group-hover:text-foreground'
                    }`}
                  />
                  <AnimatePresence>
                    {!isCollapsed && (
                      <motion.span
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -10 }}
                        className={`text-sm font-medium transition-colors ${
                          active ? 'text-emerald-400' : 'text-foreground group-hover:text-foreground'
                        }`}
                      >
                        {label}
                      </motion.span>
                    )}
                  </AnimatePresence>
                </motion.div>
              </Link>
            )
          })}
        </div>
      </nav>

      {/* Footer */}
      <div className="border-t border-border/50 p-3">
        <div
          className={`text-xs font-mono text-muted-foreground transition-all ${
            isCollapsed ? 'text-center' : ''
          }`}
        >
          {!isCollapsed && <div className="truncate">Aether v{pkg.version}</div>}
          {isCollapsed && <div>v{pkg.version.split('.')[0]}</div>}
        </div>
      </div>
    </motion.aside>
  )
}

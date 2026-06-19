'use client'

import { motion } from 'framer-motion'
import { ReactNode } from 'react'

interface GlassmorphicTooltipProps {
  content: string | ReactNode
  children: ReactNode
  position?: 'top' | 'bottom' | 'left' | 'right'
}

export function GlassmorphicTooltip({
  content,
  children,
  position = 'top',
}: GlassmorphicTooltipProps) {
  const positionClasses = {
    top: 'bottom-full mb-2 left-1/2 -translate-x-1/2',
    bottom: 'top-full mt-2 left-1/2 -translate-x-1/2',
    left: 'right-full mr-2 top-1/2 -translate-y-1/2',
    right: 'left-full ml-2 top-1/2 -translate-y-1/2',
  }

  return (
    <div className="relative group inline-block">
      {children}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        whileHover={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.2 }}
        className={`absolute ${positionClasses[position]} glassmorphic-subtle px-3 py-2 rounded-md text-xs font-mono text-foreground pointer-events-none opacity-0 group-hover:opacity-100 group-hover:pointer-events-auto transition-opacity whitespace-nowrap z-50`}
      >
        {content}
      </motion.div>
    </div>
  )
}

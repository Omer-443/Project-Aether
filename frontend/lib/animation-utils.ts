import { Variants } from 'framer-motion'

export const heavySpring = {
  type: 'spring' as const,
  stiffness: 100,
  damping: 20,
  mass: 1.2,
}

export const pageTransition: Variants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
}

export const staggerContainer: Variants = {
  animate: {
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
    },
  },
}

export const staggerItem: Variants = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0 },
}

export const cardHover = {
  whileHover: { y: -4, transition: { ...heavySpring } },
  whileTap: { scale: 0.98 },
}

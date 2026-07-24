"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface DropdownProps {
  trigger: React.ReactNode;
  children: React.ReactNode;
  align?: "left" | "right";
  className?: string;
}

export const Dropdown: React.FC<DropdownProps> = ({
  trigger,
  children,
  align = "right",
  className = ""
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="relative inline-block text-left" ref={containerRef}>
      <div onClick={() => setIsOpen((prev) => !prev)} className="cursor-pointer">
        {trigger}
      </div>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -4 }}
            transition={{ duration: 0.15 }}
            className={`absolute ${
              align === "right" ? "right-0" : "left-0"
            } mt-2 w-56 bg-[#111318] border border-[#27272A] rounded-xl shadow-2xl p-1.5 z-50 divide-y divide-[#27272A] ${className}`}
          >
            <div onClick={() => setIsOpen(false)}>{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export const DropdownItem: React.FC<{
  children: React.ReactNode;
  icon?: React.ReactNode;
  shortcut?: string;
  onClick?: () => void;
  danger?: boolean;
}> = ({ children, icon, shortcut, onClick, danger = false }) => (
  <button
    onClick={onClick}
    className={`w-full flex items-center justify-between px-3 py-2 text-xs rounded-lg transition-colors cursor-pointer ${
      danger
        ? "text-[#EF4444] hover:bg-[#EF4444]/15"
        : "text-[#94A3B8] hover:bg-[#171A21] hover:text-[#F8FAFC]"
    }`}
  >
    <div className="flex items-center gap-2.5">
      {icon && <span className={danger ? "text-[#EF4444]" : "text-[#8B5CF6]"}>{icon}</span>}
      <span>{children}</span>
    </div>
    {shortcut && <span className="text-[10px] font-mono text-[#64748B]">{shortcut}</span>}
  </button>
);

export const DropdownHeader: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="px-3 py-1.5 text-[10px] font-bold text-[#64748B] uppercase tracking-wider">
    {children}
  </div>
);

export const DropdownDivider: React.FC = () => <div className="my-1 border-t border-[#27272A]" />;

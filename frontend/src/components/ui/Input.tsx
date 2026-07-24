import React from "react";

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
  leftIcon?: React.ReactNode;
}

export const Label: React.FC<{ children: React.ReactNode; htmlFor?: string; className?: string }> = ({
  children,
  htmlFor,
  className = ""
}) => (
  <label htmlFor={htmlFor} className={`text-xs font-semibold text-[#94A3B8] block mb-1 ${className}`}>
    {children}
  </label>
);

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, helperText, leftIcon, className = "", ...props }, ref) => {
    return (
      <div className="flex flex-col gap-1 w-full">
        {label && <Label>{label}</Label>}
        <div className="relative flex items-center">
          {leftIcon && <div className="absolute left-3 text-[#64748B] pointer-events-none">{leftIcon}</div>}
          <input
            ref={ref}
            className={`w-full bg-[#0A0A0C] border border-[#2A2A35] rounded-lg px-3.5 py-2 text-xs text-[#F8FAFC] placeholder-[#64748B] focus:outline-none focus:border-[#E11D48] focus:ring-1 focus:ring-[#E11D48] transition-all ${
              leftIcon ? "pl-9" : ""
            } ${error ? "border-[#EF4444]" : ""} ${className}`}
            {...props}
          />
        </div>
        {error && <span className="text-[11px] text-[#EF4444] mt-0.5">{error}</span>}
        {helperText && !error && <span className="text-[11px] text-[#64748B] mt-0.5">{helperText}</span>}
      </div>
    );
  }
);
Input.displayName = "Input";

export interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  options: { value: string | number; label: string }[];
  error?: string;
}

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ label, options, error, className = "", ...props }, ref) => {
    return (
      <div className="flex flex-col gap-1 w-full">
        {label && <Label>{label}</Label>}
        <select
          ref={ref}
          className={`w-full bg-[#0A0A0C] border border-[#2A2A35] rounded-lg px-3.5 py-2 text-xs text-[#F8FAFC] focus:outline-none focus:border-[#E11D48] focus:ring-1 focus:ring-[#E11D48] transition-all cursor-pointer ${
            error ? "border-[#EF4444]" : ""
          } ${className}`}
          {...props}
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value} className="bg-[#121216] text-[#F8FAFC]">
              {opt.label}
            </option>
          ))}
        </select>
        {error && <span className="text-[11px] text-[#EF4444] mt-0.5">{error}</span>}
      </div>
    );
  }
);
Select.displayName = "Select";

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
}

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ label, error, className = "", ...props }, ref) => {
    return (
      <div className="flex flex-col gap-1 w-full">
        {label && <Label>{label}</Label>}
        <textarea
          ref={ref}
          className={`w-full bg-[#0A0A0C] border border-[#2A2A35] rounded-lg px-3.5 py-2 text-xs text-[#F8FAFC] placeholder-[#64748B] focus:outline-none focus:border-[#E11D48] focus:ring-1 focus:ring-[#E11D48] transition-all ${
            error ? "border-[#EF4444]" : ""
          } ${className}`}
          {...props}
        />
        {error && <span className="text-[11px] text-[#EF4444] mt-0.5">{error}</span>}
      </div>
    );
  }
);
Textarea.displayName = "Textarea";

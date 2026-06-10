import * as React from "react";

import { cn } from "@/lib/utils";

/**
 * A minimal styled native `<select>`. We use the native control rather than a Radix
 * popover to keep the dependency surface small (the rest of the UI does the same); it is
 * accessible out of the box and fine for short option lists like the model dropdown.
 */
const Select = React.forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(({ className, children, ...props }, ref) => (
  <select
    ref={ref}
    className={cn(
      "flex h-8 w-full cursor-pointer rounded-md border border-input bg-background px-2 py-1 text-xs shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
      className
    )}
    {...props}
  >
    {children}
  </select>
));
Select.displayName = "Select";

export { Select };
